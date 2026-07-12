//! Calibration 文件验证与确定性模拟发布。
//!
//! 模拟路径复用正式分析器和优化器，但输出始终标记为模拟证据，不代表真实硬件测量。

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};

use edgefit_calibration::{
    parse_evidence, parse_simulation_scenario, render_evidence_json, render_verification_json,
    render_verification_markdown, sha256_hex, verify, Attachment, Bindings, Capture, CheckStatus,
    Environment, Evidence, ExpectedBindings, Identity, LoadedAttachment, Measurements,
    RuntimeResult, Thresholds, Verification, VerificationBudget, MAX_ATTACHMENT_BYTES,
    SIMULATION_TRACE_SCHEMA,
};
use edgefit_analyze::analyze;
use edgefit_ir::{
    escape_json, load_cli_adapter_output, load_normalized_model, parse_cli_adapter_output,
    parse_normalized_model, EdgeFitResult, NormalizedModel,
};
use edgefit_optimize::optimize;
use edgefit_policy::evaluate;
use edgefit_target::parse_profile;

const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 1 << 30;
const MAX_TARGET_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SCENARIO_BYTES: u64 = 16 * 1024 * 1024;
const PPM_DENOMINATOR: u64 = 1_000_000;

const RUNTIME_FILE: &str = "simulator-runtime.bin";
const TRACE_FILE: &str = "simulation-trace.json";
const EVIDENCE_FILE: &str = "evidence.json";
const VERIFICATION_JSON_FILE: &str = "verification.json";
const VERIFICATION_MARKDOWN_FILE: &str = "verification.md";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationSimulationResult {
    pub status: String,
    pub verification_json: String,
}

pub fn verify_calibration_files(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
) -> EdgeFitResult<Verification> {
    let evidence_path = evidence_path.as_ref();
    let model_path = model_path.as_ref();
    let target_path = target_path.as_ref();

    let evidence_bytes = read_bounded_regular_file(evidence_path, MAX_EVIDENCE_BYTES, "evidence")?;
    let model_bytes = read_bounded_regular_file(model_path, MAX_MODEL_BYTES, "model")?;
    let target_bytes = read_bounded_regular_file(target_path, MAX_TARGET_BYTES, "target profile")?;

    let evidence_text = std::str::from_utf8(&evidence_bytes)
        .map_err(|err| format!("calibration evidence is not UTF-8: {err}"))?;
    let target_text = std::str::from_utf8(&target_bytes)
        .map_err(|err| format!("target profile is not UTF-8: {err}"))?;
    let evidence = parse_evidence(evidence_text)
        .map_err(|err| format!("invalid calibration evidence: {err}"))?;
    let profile = parse_profile(target_text, target_path.to_path_buf())?;
    profile.validate()?;

    if evidence.identity.target_id != profile.target_id {
        return Err(format!(
            "calibration target identity mismatch: evidence target_id {:?}, profile target_id {:?}",
            evidence.identity.target_id, profile.target_id
        ));
    }
    let arena_bytes = profile
        .peak_activation_budget_bytes
        .ok_or_else(|| "target profile has no peak activation budget".to_string())?;
    if evidence.thresholds.arena_budget_bytes != arena_bytes {
        return Err(format!(
            "calibration arena budget mismatch: evidence has {}, target profile has {}",
            evidence.thresholds.arena_budget_bytes, arena_bytes
        ));
    }

    // Calibration evidence v1 has no target FNV field. Its target ID and the SHA-256 of the
    // exact profile bytes are bound below; bind `profile.fingerprint` here when the schema grows
    // an explicit target fingerprint field.
    let _target_fingerprint_seam = &profile.fingerprint;

    let evidence_parent = evidence_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let attachment_bytes = load_declared_attachments(evidence_parent, &evidence)?;
    let loaded_attachments = evidence
        .attachments
        .iter()
        .zip(attachment_bytes.iter())
        .map(|(attachment, bytes)| LoadedAttachment {
            path: attachment.path.as_str(),
            bytes: bytes.as_slice(),
        })
        .collect::<Vec<_>>();

    let runtime_matches = evidence
        .attachments
        .iter()
        .zip(attachment_bytes.iter())
        .filter(|(attachment, _)| attachment.sha256 == evidence.bindings.runtime_binary_sha256)
        .collect::<Vec<_>>();
    let runtime_binary_sha256 = match runtime_matches.as_slice() {
        [(_, bytes)] => sha256_hex(bytes),
        [] => {
            return Err(
                "runtime binary binding does not identify a declared attachment".to_string(),
            )
        }
        _ => {
            return Err(
                "runtime binary binding identifies multiple declared attachments".to_string(),
            )
        }
    };

    let model_sha256 = sha256_hex(&model_bytes);
    let target_profile_sha256 = sha256_hex(&target_bytes);
    if evidence.bindings.model_sha256 != model_sha256 {
        return Err("calibration model SHA-256 binding mismatch".to_string());
    }
    if evidence.bindings.target_profile_sha256 != target_profile_sha256 {
        return Err("calibration target profile SHA-256 binding mismatch".to_string());
    }
    if evidence.bindings.runtime_binary_sha256 != runtime_binary_sha256 {
        return Err("calibration runtime binary SHA-256 binding mismatch".to_string());
    }
    for (attachment, bytes) in evidence.attachments.iter().zip(attachment_bytes.iter()) {
        let actual_bytes = u64::try_from(bytes.len())
            .map_err(|_| format!("attachment {} length cannot be represented", attachment.path))?;
        if actual_bytes != attachment.bytes {
            return Err(format!(
                "calibration attachment {} byte count mismatch",
                attachment.path
            ));
        }
        if sha256_hex(bytes) != attachment.sha256 {
            return Err(format!(
                "calibration attachment {} SHA-256 mismatch",
                attachment.path
            ));
        }
    }

    let expected = ExpectedBindings {
        model_sha256,
        target_profile_sha256,
        runtime_binary_sha256,
    };
    // Arena authority comes from the target profile. Evidence supplies only the latency budget.
    let budget = VerificationBudget {
        arena_bytes,
        p95_latency_ns: evidence.thresholds.p95_latency_budget_ns,
    };
    verify(&evidence, &expected, &budget, &loaded_attachments)
        .map_err(|err| format!("failed to verify calibration evidence: {err}"))
}

pub fn render_calibration_files_with_status(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    format: &str,
) -> EdgeFitResult<(String, String)> {
    if !matches!(format, "json" | "markdown") {
        return Err("calibration format must be json or markdown".to_string());
    }
    let verification = verify_calibration_files(evidence_path, model_path, target_path)?;
    let status = if verification.status == edgefit_calibration::CheckStatus::Fail {
        "fail"
    } else {
        "pass"
    };
    let rendered = match format {
        "json" => render_verification_json(&verification),
        "markdown" => render_verification_markdown(&verification),
        _ => unreachable!("format was checked above"),
    };
    Ok((status.to_string(), rendered))
}

pub fn render_calibration_files(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    format: &str,
) -> EdgeFitResult<String> {
    render_calibration_files_with_status(evidence_path, model_path, target_path, format)
        .map(|(_, rendered)| rendered)
}

/// 使用规范化模型文件生成确定性模拟证据，并在目录发布前完成 v1 验证。
pub fn simulate_calibration_files(
    source_model_path: impl AsRef<Path>,
    normalized_model_path: impl AsRef<Path>,
    adapter_generated: bool,
    target_path: impl AsRef<Path>,
    scenario_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> EdgeFitResult<CalibrationSimulationResult> {
    let model = if adapter_generated {
        load_cli_adapter_output(normalized_model_path)?
    } else {
        load_normalized_model(normalized_model_path)?
    };
    simulate_calibration_model(
        model,
        source_model_path.as_ref(),
        target_path.as_ref(),
        scenario_path.as_ref(),
        out_dir.as_ref(),
    )
}

/// 供 Python 绑定复用内存中的规范化模型；模型绑定仍覆盖原始输入文件的精确字节。
pub fn simulate_calibration_text(
    model_text: &str,
    adapter_generated: bool,
    source_model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    scenario_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> EdgeFitResult<CalibrationSimulationResult> {
    let model = if adapter_generated {
        parse_cli_adapter_output(model_text)?
    } else {
        parse_normalized_model(model_text)?
    };
    simulate_calibration_model(
        model,
        source_model_path.as_ref(),
        target_path.as_ref(),
        scenario_path.as_ref(),
        out_dir.as_ref(),
    )
}

fn simulate_calibration_model(
    model: NormalizedModel,
    source_model_path: &Path,
    target_path: &Path,
    scenario_path: &Path,
    out_dir: &Path,
) -> EdgeFitResult<CalibrationSimulationResult> {
    let model_bytes = read_bounded_regular_file(source_model_path, MAX_MODEL_BYTES, "model")?;
    let target_bytes = read_bounded_regular_file(target_path, MAX_TARGET_BYTES, "target profile")?;
    let scenario_bytes = read_bounded_regular_file(scenario_path, MAX_SCENARIO_BYTES, "scenario")?;
    let target_text = std::str::from_utf8(&target_bytes)
        .map_err(|error| format!("target profile is not UTF-8: {error}"))?;
    let scenario_text = std::str::from_utf8(&scenario_bytes)
        .map_err(|error| format!("calibration simulation scenario is not UTF-8: {error}"))?;
    let profile = parse_profile(target_text, target_path.to_path_buf())?;
    profile.validate()?;
    let scenario = parse_simulation_scenario(scenario_text)
        .map_err(|error| format!("invalid calibration simulation scenario: {error}"))?;

    let metrics = analyze(&model, &profile);
    let policy = evaluate(&model, &profile, &metrics);
    if policy.status != "pass" {
        return Err("calibration simulation requires a passing static verification".to_string());
    }
    if metrics.activation_planning_overflowed || metrics.unresolved_tensor_size_count != 0 {
        return Err("calibration simulation requires a complete activation arena plan".to_string());
    }
    let plan = optimize(&model, &profile)?;
    if plan.status != "pass" || !plan.blockers.is_empty() {
        return Err(
            "calibration simulation requires an optimization plan without blockers".to_string(),
        );
    }
    let predicted_latency_ns = plan
        .proposed_latency_ns
        .filter(|latency| *latency > 0)
        .ok_or_else(|| "calibration simulation requires a positive proposed latency".to_string())?;
    let arena_budget_bytes = profile
        .peak_activation_budget_bytes
        .ok_or_else(|| "target profile has no peak activation budget".to_string())?;
    let simulated_arena_bytes = metrics
        .planned_activation_arena_bytes
        .checked_add(scenario.arena_overhead_bytes)
        .ok_or_else(|| "calibration simulated arena overflow".to_string())?;
    let latency_samples = scenario
        .latency_scale_ppm
        .iter()
        .map(|scale| scale_ppm_half_up(predicted_latency_ns, *scale, "latency sample"))
        .collect::<EdgeFitResult<Vec<_>>>()?;
    if latency_samples.contains(&0) {
        return Err("calibration simulated latency samples must be positive".to_string());
    }
    let p95_budget_ns = scale_ppm_half_up(
        predicted_latency_ns,
        scenario.p95_budget_scale_ppm,
        "p95 latency budget",
    )?;
    if p95_budget_ns == 0 {
        return Err("calibration simulated p95 budget must be positive".to_string());
    }
    let simulated_p95_ns = nearest_rank_95(&latency_samples)?;
    let model_sha256 = sha256_hex(&model_bytes);
    let target_sha256 = sha256_hex(&target_bytes);
    let scenario_sha256 = sha256_hex(&scenario_bytes);
    let runtime_bytes = render_simulator_runtime(
        &scenario.scenario_id,
        &model_sha256,
        &target_sha256,
        &scenario_sha256,
        &profile.fingerprint,
        &plan.plan_hash,
    );
    let runtime_sha256 = sha256_hex(&runtime_bytes);
    let trace = render_simulation_trace(SimulationTraceInput {
        scenario_id: &scenario.scenario_id,
        model_sha256: &model_sha256,
        target_sha256: &target_sha256,
        scenario_sha256: &scenario_sha256,
        target_fingerprint: &profile.fingerprint,
        plan_hash: &plan.plan_hash,
        segment_count: plan.segments.len(),
        transfer_bytes: plan.transfer_bytes,
        spill_bytes: plan.spill_bytes,
        load_count: plan.events.iter().filter(|event| event.kind == "load").count(),
        store_count: plan.events.iter().filter(|event| event.kind == "store").count(),
        spill_count: plan.events.iter().filter(|event| event.kind == "spill").count(),
        reload_count: plan.events.iter().filter(|event| event.kind == "reload").count(),
        planned_arena_bytes: metrics.planned_activation_arena_bytes,
        arena_overhead_bytes: scenario.arena_overhead_bytes,
        simulated_arena_bytes,
        predicted_latency_ns,
        latency_scale_ppm: &scenario.latency_scale_ppm,
        latency_samples: &latency_samples,
        p95_budget_scale_ppm: scenario.p95_budget_scale_ppm,
        p95_budget_ns,
        simulated_p95_ns,
    })?;
    let trace_sha256 = sha256_hex(trace.as_bytes());
    let evidence = Evidence {
        identity: Identity {
            target_id: profile.target_id.clone(),
            device_id: format!("simulated-{}", scenario.scenario_id),
            runtime_name: "edgefit-deterministic-simulator".to_string(),
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
        },
        environment: Environment {
            operating_system: "simulated".to_string(),
            architecture: "virtual".to_string(),
            hardware: "deterministic-host-simulator".to_string(),
            toolchain: "edgefit-rust-core".to_string(),
        },
        capture: Capture {
            captured_at: scenario.captured_at.clone(),
            command: format!("edgefit calibration simulate --scenario {}", scenario.scenario_id),
            warmup_runs: scenario.warmup_runs,
            measured_runs: u64::try_from(latency_samples.len())
                .map_err(|_| "simulation sample count cannot be represented".to_string())?,
        },
        bindings: Bindings {
            model_sha256,
            target_profile_sha256: target_sha256,
            runtime_binary_sha256: runtime_sha256.clone(),
        },
        runtime: RuntimeResult {
            accepted: true,
            rejected_reason: None,
        },
        measurements: Measurements {
            arena_high_water_bytes: simulated_arena_bytes,
            latency_ns: latency_samples,
        },
        thresholds: Thresholds {
            arena_budget_bytes,
            p95_latency_budget_ns: p95_budget_ns,
        },
        attachments: vec![
            Attachment {
                name: "simulator-runtime".to_string(),
                path: RUNTIME_FILE.to_string(),
                media_type: "application/octet-stream".to_string(),
                bytes: u64::try_from(runtime_bytes.len())
                    .map_err(|_| "simulator runtime size cannot be represented".to_string())?,
                sha256: runtime_sha256,
            },
            Attachment {
                name: "simulation-trace".to_string(),
                path: TRACE_FILE.to_string(),
                media_type: "application/json".to_string(),
                bytes: u64::try_from(trace.len())
                    .map_err(|_| "simulation trace size cannot be represented".to_string())?,
                sha256: trace_sha256,
            },
        ],
    };
    publish_simulation_directory(
        out_dir,
        source_model_path,
        target_path,
        &runtime_bytes,
        &trace,
        &render_evidence_json(&evidence),
    )
}

struct SimulationTraceInput<'a> {
    scenario_id: &'a str,
    model_sha256: &'a str,
    target_sha256: &'a str,
    scenario_sha256: &'a str,
    target_fingerprint: &'a str,
    plan_hash: &'a str,
    segment_count: usize,
    transfer_bytes: u64,
    spill_bytes: u64,
    load_count: usize,
    store_count: usize,
    spill_count: usize,
    reload_count: usize,
    planned_arena_bytes: u64,
    arena_overhead_bytes: u64,
    simulated_arena_bytes: u64,
    predicted_latency_ns: u64,
    latency_scale_ppm: &'a [u64],
    latency_samples: &'a [u64],
    p95_budget_scale_ppm: u64,
    p95_budget_ns: u64,
    simulated_p95_ns: u64,
}

fn render_simulator_runtime(
    scenario_id: &str,
    model_sha256: &str,
    target_sha256: &str,
    scenario_sha256: &str,
    target_fingerprint: &str,
    plan_hash: &str,
) -> Vec<u8> {
    format!(
        "EDGEFIT_SIMULATOR_RUNTIME_V1\nversion={}\nscenario={}\nmodel_sha256={}\ntarget_sha256={}\nscenario_sha256={}\ntarget_fingerprint={}\nplan_hash={}\nconfidence=simulated\n",
        env!("CARGO_PKG_VERSION"),
        scenario_id,
        model_sha256,
        target_sha256,
        scenario_sha256,
        target_fingerprint,
        plan_hash,
    )
    .into_bytes()
}

fn render_simulation_trace(input: SimulationTraceInput<'_>) -> EdgeFitResult<String> {
    let error_ns = i128::from(input.simulated_p95_ns)
        .checked_sub(i128::from(input.predicted_latency_ns))
        .ok_or_else(|| "simulation prediction error overflow".to_string())?;
    let error_ppm = error_ns
        .checked_mul(i128::from(PPM_DENOMINATOR))
        .and_then(|value| value.checked_div(i128::from(input.predicted_latency_ns)))
        .ok_or_else(|| "simulation prediction error ppm overflow".to_string())?;
    let scales = input
        .latency_scale_ppm
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    let samples = input
        .latency_samples
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\n  \"schema\": \"{}\",\n  \"confidence\": \"simulated\",\n  \"scenario_id\": \"{}\",\n  \"bindings\": {{\"model_sha256\":\"{}\",\"target_profile_sha256\":\"{}\",\"scenario_sha256\":\"{}\",\"target_fingerprint\":\"{}\",\"plan_hash\":\"{}\"}},\n  \"plan\": {{\"segment_count\":\"{}\",\"transfer_bytes\":\"{}\",\"spill_bytes\":\"{}\",\"events\":{{\"load\":\"{}\",\"store\":\"{}\",\"spill\":\"{}\",\"reload\":\"{}\"}}}},\n  \"arena\": {{\"planned_bytes\":\"{}\",\"overhead_bytes\":\"{}\",\"simulated_bytes\":\"{}\"}},\n  \"latency\": {{\"predicted_ns\":\"{}\",\"scale_ppm\":[{}],\"samples_ns\":[{}],\"p95_budget_scale_ppm\":\"{}\",\"p95_budget_ns\":\"{}\",\"simulated_p95_ns\":\"{}\",\"prediction_error_ns\":\"{}\",\"prediction_error_ppm\":\"{}\"}},\n  \"limitations\": [\"not_real_hardware\",\"controlled_perturbation\",\"no_device_attestation\",\"no_profile_mutation_authority\"]\n}}\n",
        SIMULATION_TRACE_SCHEMA,
        escape_json(input.scenario_id),
        input.model_sha256,
        input.target_sha256,
        input.scenario_sha256,
        escape_json(input.target_fingerprint),
        escape_json(input.plan_hash),
        input.segment_count,
        input.transfer_bytes,
        input.spill_bytes,
        input.load_count,
        input.store_count,
        input.spill_count,
        input.reload_count,
        input.planned_arena_bytes,
        input.arena_overhead_bytes,
        input.simulated_arena_bytes,
        input.predicted_latency_ns,
        scales,
        samples,
        input.p95_budget_scale_ppm,
        input.p95_budget_ns,
        input.simulated_p95_ns,
        error_ns,
        error_ppm,
    ))
}

fn scale_ppm_half_up(base: u64, scale_ppm: u64, label: &str) -> EdgeFitResult<u64> {
    let numerator = u128::from(base)
        .checked_mul(u128::from(scale_ppm))
        .and_then(|value| value.checked_add(u128::from(PPM_DENOMINATOR / 2)))
        .ok_or_else(|| format!("calibration simulated {label} overflow"))?;
    u64::try_from(numerator / u128::from(PPM_DENOMINATOR))
        .map_err(|_| format!("calibration simulated {label} cannot be represented"))
}

fn nearest_rank_95(samples: &[u64]) -> EdgeFitResult<u64> {
    if samples.is_empty() {
        return Err("calibration simulated latency samples must not be empty".to_string());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let numerator = sorted
        .len()
        .checked_mul(95)
        .and_then(|value| value.checked_add(99))
        .ok_or_else(|| "calibration simulated percentile overflow".to_string())?;
    sorted
        .get(numerator / 100 - 1)
        .copied()
        .ok_or_else(|| "calibration simulated percentile is unavailable".to_string())
}

fn publish_simulation_directory(
    out_dir: &Path,
    model_path: &Path,
    target_path: &Path,
    runtime_bytes: &[u8],
    trace: &str,
    evidence: &str,
) -> EdgeFitResult<CalibrationSimulationResult> {
    if out_dir.exists() {
        return Err(format!(
            "calibration simulation output directory already exists: {}",
            out_dir.display()
        ));
    }
    let parent = out_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create simulation output parent: {error}"))?;
    let name = out_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("calibration simulation output directory requires a UTF-8 name")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("system clock error: {error}"))?
        .as_nanos();
    let temporary = parent.join(format!(
        ".{name}.edgefit-simulation-{}-{stamp}.tmp",
        std::process::id()
    ));
    fs::create_dir(&temporary)
        .map_err(|error| format!("failed to create temporary simulation directory: {error}"))?;
    let result: EdgeFitResult<CalibrationSimulationResult> = (|| {
        write_synced_file(&temporary.join(RUNTIME_FILE), runtime_bytes)?;
        write_synced_file(&temporary.join(TRACE_FILE), trace.as_bytes())?;
        write_synced_file(&temporary.join(EVIDENCE_FILE), evidence.as_bytes())?;
        let verification = verify_calibration_files(
            temporary.join(EVIDENCE_FILE),
            model_path,
            target_path,
        )?;
        let verification_json = render_verification_json(&verification);
        write_synced_file(
            &temporary.join(VERIFICATION_JSON_FILE),
            verification_json.as_bytes(),
        )?;
        write_synced_file(
            &temporary.join(VERIFICATION_MARKDOWN_FILE),
            render_verification_markdown(&verification).as_bytes(),
        )?;
        fs::rename(&temporary, out_dir)
            .map_err(|error| format!("failed to publish simulation directory: {error}"))?;
        Ok(CalibrationSimulationResult {
            status: if verification.status == CheckStatus::Pass {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            verification_json,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> EdgeFitResult<()> {
    use std::io::Write;
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create simulation file {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("failed to write simulation file {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync simulation file {}: {error}", path.display()))
}

fn load_declared_attachments(
    evidence_parent: &Path,
    evidence: &edgefit_calibration::Evidence,
) -> EdgeFitResult<Vec<Vec<u8>>> {
    let parent = fs::canonicalize(evidence_parent).map_err(|err| {
        format!(
            "failed to canonicalize evidence directory {}: {err}",
            evidence_parent.display()
        )
    })?;
    if !fs::metadata(&parent)
        .map_err(|err| format!("failed to inspect evidence directory: {err}"))?
        .is_dir()
    {
        return Err("evidence parent is not a directory".to_string());
    }

    let mut paths = BTreeSet::new();
    let mut loaded = Vec::with_capacity(evidence.attachments.len());
    for attachment in &evidence.attachments {
        let relative = safe_relative_attachment_path(&attachment.path)?;
        if !paths.insert(relative.clone()) {
            return Err(format!("duplicate attachment path {}", attachment.path));
        }
        if attachment.bytes > MAX_ATTACHMENT_BYTES {
            return Err(format!(
                "attachment {} exceeds byte limit",
                attachment.path
            ));
        }

        let path = parent.join(&relative);
        reject_symlink_components(&parent, &relative)?;
        let canonical = fs::canonicalize(&path).map_err(|err| {
            format!("failed to resolve attachment {}: {err}", attachment.path)
        })?;
        if !canonical.starts_with(&parent) {
            return Err(format!(
                "attachment {} escapes the evidence directory",
                attachment.path
            ));
        }
        let bytes = read_bounded_regular_file(&canonical, MAX_ATTACHMENT_BYTES, "attachment")?;
        loaded.push(bytes);
    }
    Ok(loaded)
}

fn safe_relative_attachment_path(value: &str) -> EdgeFitResult<PathBuf> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("unsafe attachment path {value:?}"));
    }
    let mut safe = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) if !part.is_empty() => safe.push(part),
            _ => return Err(format!("unsafe attachment path {value:?}")),
        }
    }
    if safe.as_os_str().is_empty() {
        return Err(format!("unsafe attachment path {value:?}"));
    }
    Ok(safe)
}

fn reject_symlink_components(parent: &Path, relative: &Path) -> EdgeFitResult<()> {
    let mut current = parent.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(format!(
                "unsafe attachment path {}",
                relative.display()
            ));
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "attachment path contains a symlink: {}",
                    relative.display()
                ));
            }
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Err(format!(
                    "attachment path does not exist: {}",
                    relative.display()
                ));
            }
            Err(err) => {
                return Err(format!(
                    "failed to inspect attachment path {}: {err}",
                    relative.display()
                ));
            }
        }
    }
    Ok(())
}

fn read_bounded_regular_file(path: &Path, max_bytes: u64, label: &str) -> EdgeFitResult<Vec<u8>> {
    let link_metadata = fs::symlink_metadata(path)
        .map_err(|err| format!("failed to inspect {label} {}: {err}", path.display()))?;
    if link_metadata.file_type().is_symlink() {
        return Err(format!("{label} {} must not be a symlink", path.display()));
    }
    if !link_metadata.file_type().is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    if link_metadata.len() > max_bytes {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }

    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {label} {}: {err}", path.display()))?;
    let open_metadata = file
        .metadata()
        .map_err(|err| format!("failed to inspect open {label} {}: {err}", path.display()))?;
    if !open_metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    if open_metadata.len() > max_bytes {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }

    let capacity = usize::try_from(open_metadata.len())
        .map_err(|_| format!("{label} size cannot be represented"))?;
    let take_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} byte limit overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|err| format!("failed to allocate buffer for {label}: {err}"))?;
    file.by_ref()
        .take(take_limit)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read {label} {}: {err}", path.display()))?;
    if u64::try_from(bytes.len()).map_err(|_| format!("{label} size cannot be represented"))?
        > max_bytes
    {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }
    if u64::try_from(bytes.len()).map_err(|_| format!("{label} size cannot be represented"))?
        != open_metadata.len()
    {
        return Err(format!("{label} {} changed while being read", path.display()));
    }
    Ok(bytes)
}
