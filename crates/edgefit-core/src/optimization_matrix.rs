//! 多 target profile 的确定性优化鲁棒性矩阵。
//!
//! 本模块只编排完整 profile，不修改 seed 参数，也不把跨 profile 的模拟代价解释为实机测量。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use edgefit_calibration::sha256_hex;
use edgefit_ir::{
    escape_json, load_cli_adapter_output, load_normalized_model, parse_cli_adapter_output,
    parse_json, parse_normalized_model, EdgeFitResult, JsonValue, NormalizedModel,
};
use edgefit_optimize::{optimize, render_plan, OptimizationPlan};
use edgefit_target::load_profile;

const MATRIX_SCHEMA: &str = "edgefit.optimizer_profile_matrix.v1";
const RESULT_SCHEMA: &str = "edgefit.optimization_matrix.v1";
const MAX_MATRIX_BYTES: u64 = 1024 * 1024;
const MAX_PROFILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MATRIX_PROFILES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizationMatrixCase {
    pub id: String,
    pub profile_path: String,
    pub plan: OptimizationPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizationMatrix {
    pub schema: String,
    pub status: String,
    pub classification: String,
    pub matrix_id: String,
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub stable_assignments: bool,
    pub modeled_latency_min_ns: Option<u64>,
    pub modeled_latency_max_ns: Option<u64>,
    pub cases: Vec<OptimizationMatrixCase>,
    pub matrix_hash: String,
}

#[derive(Clone, Debug)]
struct ProfileSpec {
    id: String,
    path: PathBuf,
    display_path: String,
}

pub fn optimize_matrix_files(
    normalized_model_path: impl AsRef<Path>,
    adapter_generated: bool,
    manifest_path: impl AsRef<Path>,
) -> EdgeFitResult<OptimizationMatrix> {
    let model = if adapter_generated {
        load_cli_adapter_output(normalized_model_path)?
    } else {
        load_normalized_model(normalized_model_path)?
    };
    optimize_matrix_model(model, manifest_path.as_ref())
}

pub fn optimize_matrix_text(
    model_text: &str,
    adapter_generated: bool,
    manifest_path: impl AsRef<Path>,
) -> EdgeFitResult<OptimizationMatrix> {
    let model = if adapter_generated {
        parse_cli_adapter_output(model_text)?
    } else {
        parse_normalized_model(model_text)?
    };
    optimize_matrix_model(model, manifest_path.as_ref())
}

fn optimize_matrix_model(
    model: NormalizedModel,
    manifest_path: &Path,
) -> EdgeFitResult<OptimizationMatrix> {
    let manifest_bytes = read_bounded_regular_file(manifest_path, MAX_MATRIX_BYTES, "matrix")?;
    let manifest_text = std::str::from_utf8(&manifest_bytes)
        .map_err(|error| format!("optimizer matrix is not UTF-8: {error}"))?;
    let (matrix_id, profile_specs) = parse_manifest(manifest_text, manifest_path)?;
    let mut cases = Vec::with_capacity(profile_specs.len());
    for spec in profile_specs {
        let profile = load_profile(&spec.path)?;
        let plan = optimize(&model, &profile)?;
        cases.push(OptimizationMatrixCase {
            id: spec.id,
            profile_path: spec.display_path,
            plan,
        });
    }

    let passing = cases.iter().filter(|case| case.plan.status == "pass").count();
    let classification = if passing == cases.len() {
        "robust_pass"
    } else if passing == 0 {
        "robust_fail"
    } else {
        "fragile"
    };
    let first_assignments = cases.first().map(assignment_signature);
    let stable_assignments = first_assignments.as_ref().is_some_and(|first| {
        cases
            .iter()
            .all(|case| assignment_signature(case).as_slice() == first.as_slice())
    });
    let latencies = cases
        .iter()
        .filter_map(|case| case.plan.proposed_latency_ns)
        .collect::<Vec<_>>();
    let modeled_latency_min_ns = latencies.iter().copied().min();
    let modeled_latency_max_ns = latencies.iter().copied().max();
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let mut matrix = OptimizationMatrix {
        schema: RESULT_SCHEMA.to_string(),
        status: if classification == "robust_pass" {
            "pass"
        } else {
            "fail"
        }
        .to_string(),
        classification: classification.to_string(),
        matrix_id,
        manifest_sha256,
        model_sha256: model.sha256,
        stable_assignments,
        modeled_latency_min_ns,
        modeled_latency_max_ns,
        cases,
        matrix_hash: String::new(),
    };
    matrix.matrix_hash = sha256_hex(matrix_hash_payload(&matrix).as_bytes());
    Ok(matrix)
}

fn assignment_signature(case: &OptimizationMatrixCase) -> Vec<(String, Option<String>, Option<String>)> {
    case.plan
        .assignments
        .iter()
        .map(|assignment| {
            (
                assignment.device.clone(),
                assignment.kernel_id.clone(),
                assignment.recipe_id.clone(),
            )
        })
        .collect()
}

fn parse_manifest(input: &str, manifest_path: &Path) -> EdgeFitResult<(String, Vec<ProfileSpec>)> {
    let root = parse_json(input)?;
    let object = root
        .as_object()
        .ok_or("optimizer matrix manifest must be an object")?;
    exact_fields(object, &["schema", "matrix_id", "profiles"], "optimizer matrix")?;
    if required_string(object, "schema")? != MATRIX_SCHEMA {
        return Err(format!("optimizer matrix schema must be {MATRIX_SCHEMA}"));
    }
    let matrix_id = safe_id(&required_string(object, "matrix_id")?, "matrix_id")?;
    let profiles = object
        .get("profiles")
        .and_then(JsonValue::as_array)
        .ok_or("optimizer matrix profiles must be an array")?;
    if profiles.is_empty() || profiles.len() > MAX_MATRIX_PROFILES {
        return Err(format!(
            "optimizer matrix profiles must contain between 1 and {MAX_MATRIX_PROFILES} entries"
        ));
    }
    let base = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut ids = BTreeSet::new();
    let mut canonical_paths = BTreeSet::new();
    let mut specs = Vec::with_capacity(profiles.len());
    for value in profiles {
        let profile = value
            .as_object()
            .ok_or("optimizer matrix profile entry must be an object")?;
        exact_fields(profile, &["id", "path"], "optimizer matrix profile")?;
        let id = safe_id(&required_string(profile, "id")?, "profile id")?;
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate optimizer matrix profile id {id}"));
        }
        let declared = required_string(profile, "path")?;
        let relative = safe_relative_path(&declared)?;
        let path = base.join(relative);
        read_bounded_regular_file(&path, MAX_PROFILE_BYTES, "target profile")?;
        let canonical = fs::canonicalize(&path)
            .map_err(|error| format!("failed to resolve target profile {}: {error}", path.display()))?;
        if !canonical_paths.insert(canonical) {
            return Err(format!("duplicate optimizer matrix profile path {declared}"));
        }
        specs.push(ProfileSpec {
            id,
            path,
            display_path: declared,
        });
    }
    Ok((matrix_id, specs))
}

fn exact_fields(
    object: &std::collections::BTreeMap<String, JsonValue>,
    expected: &[&str],
    context: &str,
) -> EdgeFitResult<()> {
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    for field in object.keys() {
        if !expected.contains(field.as_str()) {
            return Err(format!("unknown {context} field {field}"));
        }
    }
    for field in expected {
        if !object.contains_key(field) {
            return Err(format!("missing {context} field {field}"));
        }
    }
    Ok(())
}

fn required_string(
    object: &std::collections::BTreeMap<String, JsonValue>,
    field: &str,
) -> EdgeFitResult<String> {
    object
        .get(field)
        .and_then(JsonValue::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("optimizer matrix {field} must be a non-empty string"))
}

fn safe_id(value: &str, field: &str) -> EdgeFitResult<String> {
    if value.len() > 128
        || !value
            .bytes()
            .enumerate()
            .all(|(index, byte)| byte.is_ascii_alphanumeric() || (index != 0 && matches!(byte, b'.' | b'_' | b'-')))
    {
        return Err(format!("optimizer matrix {field} is not a safe identifier"));
    }
    Ok(value.to_string())
}

fn safe_relative_path(value: &str) -> EdgeFitResult<PathBuf> {
    let path = Path::new(value);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::Prefix(_) | Component::RootDir))
    {
        return Err("optimizer matrix profile path must be a safe relative path".to_string());
    }
    Ok(path.to_path_buf())
}

fn read_bounded_regular_file(path: &Path, limit: u64, label: &str) -> EdgeFitResult<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() > limit {
        return Err(format!("{label} exceeds byte limit {limit}"));
    }
    fs::read(path).map_err(|error| format!("failed to read {label} {}: {error}", path.display()))
}

fn matrix_hash_payload(matrix: &OptimizationMatrix) -> String {
    let plans = matrix
        .cases
        .iter()
        .map(|case| format!("{}:{}", case.id, case.plan.plan_hash))
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "schema={};status={};class={};matrix={};manifest={};model={};stable={};min={:?};max={:?};plans={plans}",
        matrix.schema,
        matrix.status,
        matrix.classification,
        matrix.matrix_id,
        matrix.manifest_sha256,
        matrix.model_sha256,
        matrix.stable_assignments,
        matrix.modeled_latency_min_ns,
        matrix.modeled_latency_max_ns,
    )
}

pub fn render_optimization_matrix(matrix: &OptimizationMatrix, format: &str) -> String {
    if format == "markdown" {
        return render_markdown(matrix);
    }
    let optional = |value: Option<u64>| {
        value
            .map(|item| item.to_string())
            .unwrap_or_else(|| "null".to_string())
    };
    let cases = matrix
        .cases
        .iter()
        .map(|case| {
            format!(
                "{{\"id\":\"{}\",\"profile_path\":\"{}\",\"plan\":{}}}",
                escape_json(&case.id),
                escape_json(&case.profile_path),
                render_plan(&case.plan, "json")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\n  \"schema\": \"{}\",\n  \"status\": \"{}\",\n  \"classification\": \"{}\",\n  \"matrix_id\": \"{}\",\n  \"manifest_sha256\": \"{}\",\n  \"model_sha256\": \"{}\",\n  \"stable_assignments\": {},\n  \"modeled_latency_range_ns\": {{\"min\":{},\"max\":{}}},\n  \"cases\": [{}],\n  \"matrix_hash\": \"{}\"\n}}\n",
        matrix.schema,
        matrix.status,
        matrix.classification,
        escape_json(&matrix.matrix_id),
        matrix.manifest_sha256,
        escape_json(&matrix.model_sha256),
        matrix.stable_assignments,
        optional(matrix.modeled_latency_min_ns),
        optional(matrix.modeled_latency_max_ns),
        cases,
        matrix.matrix_hash,
    )
}

fn render_markdown(matrix: &OptimizationMatrix) -> String {
    let rows = matrix
        .cases
        .iter()
        .map(|case| {
            format!(
                "| {} | {} | {} | {} | {} | {} |",
                case.id,
                case.plan.target_id,
                case.plan.status,
                case.plan
                    .proposed_latency_ns
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                case.plan.spill_bytes,
                case.plan.proposed_blockers,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "# EdgeFit Optimization Matrix\n\n**Status:** `{}`  \n**Classification:** `{}`  \n**Stable assignments:** `{}`  \n**Matrix hash:** `{}`\n\n| Case | Target | Status | Modeled latency (ns) | Spill bytes | Blockers |\n| --- | --- | --- | ---: | ---: | ---: |\n{}\n\nAll costs are declared profile assumptions; this report contains no hardware measurement.\n",
        matrix.status,
        matrix.classification,
        matrix.stable_assignments,
        matrix.matrix_hash,
        rows,
    )
}
