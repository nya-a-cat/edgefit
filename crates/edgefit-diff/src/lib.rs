use edgefit_ir::{escape_json, parse_json, EdgeFitResult, JsonValue};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub edgefit_version: String,
    pub status: String,
    pub model_path: String,
    pub model_hash: String,
    pub target_id: String,
    pub target_profile_source: String,
    pub target_profile_confidence: String,
    pub target_profile_last_verified: String,
    pub target_profile_fingerprint: String,
    pub metrics: BTreeMap<String, String>,
    pub diagnostic_ids: BTreeSet<String>,
    pub diagnostic_severities: BTreeMap<String, String>,
    /// suppression 不改变诊断严重级别，仅改变其是否参与当前报告状态计算。
    pub suppressed_diagnostic_severities: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DiagnosticState {
    Active,
    Suppressed,
}

type DiagnosticKey = (String, DiagnosticState);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffReport {
    pub status: String,
    pub edgefit_version: String,
    pub target_id: String,
    pub old_status: String,
    pub new_status: String,
    pub old_model: String,
    pub new_model: String,
    pub metric_changes: BTreeMap<String, (String, String)>,
    pub added_diagnostics: Vec<String>,
    pub resolved_diagnostics: Vec<String>,
}

pub fn load_snapshot(path: &str) -> EdgeFitResult<Snapshot> {
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read snapshot: {err}"))?;
    parse_snapshot(&text)
}

pub fn parse_snapshot(text: &str) -> EdgeFitResult<Snapshot> {
    let value = parse_json(text)?;
    let obj = value.as_object().ok_or("snapshot must be a JSON object")?;
    let schema = required_str(obj, "schema")?;
    if schema != "edgefit.snapshot.v1" && schema != "edgefit.report.v1" {
        return Err(
            "snapshot must use schema edgefit.snapshot.v1 or legacy edgefit.report.v1"
                .to_string(),
        );
    }
    let edgefit_version = required_str(obj, "edgefit_version")?.to_string();
    let status = required_str(obj, "status")?.to_string();
    if !matches!(status.as_str(), "pass" | "fail") {
        return Err("snapshot status must be pass or fail".to_string());
    }
    let (model_path, model_hash, target_id, profile_source, profile_confidence, profile_last_verified, profile_fingerprint) = if schema == "edgefit.snapshot.v1" {
        (
            required_str(obj, "model_path")?.to_string(),
            required_str(obj, "model_hash")?.to_string(),
            required_str(obj, "target_id")?.to_string(),
            required_str(obj, "target_profile_source")?.to_string(),
            required_str(obj, "target_profile_confidence")?.to_string(),
            required_str(obj, "target_profile_last_verified")?.to_string(),
            required_str(obj, "target_profile_fingerprint")?.to_string(),
        )
    } else {
        let model = required_object(obj, "model")?;
        let target = required_object(obj, "target")?;
        let metadata = required_object(target, "profile_metadata")?;
        (
            required_str(model, "path")?.to_string(),
            required_str(model, "sha256")?.to_string(),
            required_str(target, "id")?.to_string(),
            required_str(metadata, "source")?.to_string(),
            required_str(metadata, "confidence")?.to_string(),
            required_str(metadata, "last_verified")?.to_string(),
            required_str(metadata, "fingerprint")?.to_string(),
        )
    };
    let metrics = required_object(obj, "metrics")?
        .iter()
        .map(|(key, value)| (key.clone(), value_to_string(value)))
        .collect::<BTreeMap<_, _>>();
    let (diagnostic_ids, diagnostic_severities) =
        parse_diagnostics(required_array(obj, "diagnostics")?, "diagnostics")?;
    let (_, suppressed_diagnostic_severities) = parse_diagnostics(
        optional_array(obj, "suppressed_diagnostics")?,
        "suppressed_diagnostics",
    )?;
    let has_error = diagnostic_severities
        .values()
        .any(|severity| severity == "error");
    if (status == "fail") != has_error {
        return Err("snapshot status is inconsistent with active error diagnostics".to_string());
    }
    Ok(Snapshot {
        edgefit_version,
        status,
        model_path,
        model_hash,
        target_id,
        target_profile_source: profile_source,
        target_profile_confidence: profile_confidence,
        target_profile_last_verified: profile_last_verified,
        target_profile_fingerprint: profile_fingerprint,
        metrics,
        diagnostic_ids,
        diagnostic_severities,
        suppressed_diagnostic_severities,
    })
}

fn parse_diagnostics(
    diagnostics: &[JsonValue],
    field_name: &str,
) -> EdgeFitResult<(BTreeSet<String>, BTreeMap<String, String>)> {
    let mut diagnostic_ids = BTreeSet::new();
    let mut diagnostic_severities = BTreeMap::new();
    let mut occurrences = BTreeMap::<String, u64>::new();
    for item in diagnostics {
        let diagnostic = item
            .as_object()
            .ok_or_else(|| format!("snapshot {field_name} must contain only objects"))?;
        let id = required_str(diagnostic, "id")?;
        let severity = required_str(diagnostic, "severity")?;
        if !matches!(severity, "error" | "warning") {
            return Err(format!("snapshot diagnostic {id} has invalid severity {severity}"));
        }
        let location = optional_str(diagnostic, "location")?;
        let base_key = location
            .map(|location| format!("{id}@{location}"))
            .unwrap_or_else(|| id.to_string());
        let occurrence = occurrences.entry(base_key.clone()).or_insert(0);
        let key = if *occurrence == 0 {
            base_key
        } else {
            format!("{base_key}#{}", *occurrence + 1)
        };
        *occurrence += 1;
        diagnostic_ids.insert(id.to_string());
        diagnostic_severities.insert(key, severity.to_string());
    }
    Ok((diagnostic_ids, diagnostic_severities))
}

pub fn diff_snapshots(old: &Snapshot, new: &Snapshot) -> EdgeFitResult<DiffReport> {
    if old.edgefit_version != new.edgefit_version {
        return Err(format!(
            "cannot compare snapshots from different EdgeFit versions: {} vs {}",
            old.edgefit_version, new.edgefit_version
        ));
    }
    if old.target_id != new.target_id
        || old.target_profile_source != new.target_profile_source
        || old.target_profile_confidence != new.target_profile_confidence
        || old.target_profile_last_verified != new.target_profile_last_verified
        || old.target_profile_fingerprint != new.target_profile_fingerprint
    {
        return Err("cannot compare snapshots from different target profile identities".to_string());
    }
    let mut metric_changes = BTreeMap::new();
    let keys = old
        .metrics
        .keys()
        .chain(new.metrics.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    for key in keys {
        let old_value = old
            .metrics
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "null".to_string());
        let new_value = new
            .metrics
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "null".to_string());
        if old_value != new_value {
            metric_changes.insert(key, (old_value, new_value));
        }
    }
    let old_diagnostic_severities = all_diagnostic_severities(old);
    let new_diagnostic_severities = all_diagnostic_severities(new);
    let old_diagnostic_keys = old_diagnostic_severities
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let new_diagnostic_keys = new_diagnostic_severities
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let added_keys = new_diagnostic_keys
        .difference(&old_diagnostic_keys)
        .cloned()
        .collect::<Vec<_>>();
    let resolved_keys = old_diagnostic_keys
        .difference(&new_diagnostic_keys)
        .cloned()
        .collect::<Vec<_>>();
    let mut added_diagnostics = added_keys
        .iter()
        .map(|key| diagnostic_label(key, &new_diagnostic_severities))
        .collect::<Vec<_>>();
    let resolved_diagnostics = resolved_keys
        .iter()
        .map(|key| diagnostic_label(key, &old_diagnostic_severities))
        .collect::<Vec<_>>();
    let mut blocking_regression = added_keys.iter().any(|key| {
        new_diagnostic_severities
            .get(key)
            .map(|severity| severity == "error")
            .unwrap_or(false)
    });
    for key in old_diagnostic_keys.intersection(&new_diagnostic_keys) {
        let old_severity = old_diagnostic_severities
            .get(key)
            .map(String::as_str)
            .unwrap_or("warning");
        let new_severity = new_diagnostic_severities
            .get(key)
            .map(String::as_str)
            .unwrap_or("warning");
        if severity_rank(new_severity) > severity_rank(old_severity) {
            added_diagnostics.push(format!(
                "{} [{old_severity} -> {new_severity}]",
                diagnostic_key_label(key)
            ));
            blocking_regression |= new_severity == "error";
        }
    }
    added_diagnostics.sort();
    Ok(DiffReport {
        status: if blocking_regression || (old.status != "fail" && new.status == "fail") {
            "fail"
        } else {
            "pass"
        }
        .to_string(),
        edgefit_version: old.edgefit_version.clone(),
        target_id: old.target_id.clone(),
        old_status: old.status.clone(),
        new_status: new.status.clone(),
        old_model: old.model_path.clone(),
        new_model: new.model_path.clone(),
        metric_changes,
        added_diagnostics,
        resolved_diagnostics,
    })
}

pub fn render_diff(diff: &DiffReport, format: &str) -> String {
    if format == "json" {
        render_diff_json(diff)
    } else {
        render_diff_markdown(diff)
    }
}

pub fn render_diff_markdown(diff: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str("# EdgeFit Snapshot Diff\n\n");
    out.push_str(&format!("**Status:** `{}`\n", diff.status));
    out.push_str(&format!("**Old report status:** `{}`\n", diff.old_status));
    out.push_str(&format!("**New report status:** `{}`\n", diff.new_status));
    out.push_str(&format!("**Old model:** `{}`\n", diff.old_model));
    out.push_str(&format!("**New model:** `{}`\n\n", diff.new_model));
    out.push_str("## Metric Changes\n\n");
    if diff.metric_changes.is_empty() {
        out.push_str("No metric changes.\n");
    } else {
        out.push_str("| Metric | Old | New |\n| --- | --- | --- |\n");
        for (key, (old_value, new_value)) in &diff.metric_changes {
            out.push_str(&format!("| `{key}` | `{old_value}` | `{new_value}` |\n"));
        }
    }
    out.push_str("\n## Diagnostic Changes\n\n");
    out.push_str(&format!(
        "Added: `{}`\n",
        list_or_none(&diff.added_diagnostics)
    ));
    out.push_str(&format!(
        "Resolved: `{}`\n",
        list_or_none(&diff.resolved_diagnostics)
    ));
    out
}

pub fn render_diff_json(diff: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": \"edgefit.diff.v1\",\n");
    out.push_str(&format!(
        "  \"status\": \"{}\",\n",
        escape_json(&diff.status)
    ));
    out.push_str(&format!(
        "  \"edgefit_version\": \"{}\",\n",
        escape_json(&diff.edgefit_version)
    ));
    out.push_str(&format!(
        "  \"target_id\": \"{}\",\n",
        escape_json(&diff.target_id)
    ));
    out.push_str(&format!(
        "  \"old_status\": \"{}\",\n",
        escape_json(&diff.old_status)
    ));
    out.push_str(&format!(
        "  \"new_status\": \"{}\",\n",
        escape_json(&diff.new_status)
    ));
    out.push_str(&format!(
        "  \"old_model\": \"{}\",\n",
        escape_json(&diff.old_model)
    ));
    out.push_str(&format!(
        "  \"new_model\": \"{}\",\n",
        escape_json(&diff.new_model)
    ));
    out.push_str("  \"metric_changes\": {\n");
    for (index, (key, (old_value, new_value))) in diff.metric_changes.iter().enumerate() {
        out.push_str(&format!(
            "    \"{}\": {{ \"old\": \"{}\", \"new\": \"{}\" }}",
            escape_json(key),
            escape_json(old_value),
            escape_json(new_value)
        ));
        if index + 1 != diff.metric_changes.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  },\n");
    out.push_str(&format!(
        "  \"added_diagnostics\": {},\n",
        json_string_array(&diff.added_diagnostics)
    ));
    out.push_str(&format!(
        "  \"resolved_diagnostics\": {}\n",
        json_string_array(&diff.resolved_diagnostics)
    ));
    out.push_str("}\n");
    out
}

fn get_str<'a>(obj: &'a BTreeMap<String, JsonValue>, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(JsonValue::as_str)
}

fn required_str<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a str> {
    get_str(obj, key).ok_or_else(|| format!("snapshot is missing string field {key}"))
}

fn optional_str<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<Option<&'a str>> {
    match obj.get(key) {
        None => Ok(None),
        Some(JsonValue::String(value)) => Ok(Some(value)),
        Some(_) => Err(format!("snapshot field {key} must be a string")),
    }
}

fn required_object<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a BTreeMap<String, JsonValue>> {
    obj.get(key)
        .and_then(JsonValue::as_object)
        .ok_or_else(|| format!("snapshot is missing object field {key}"))
}

fn required_array<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a [JsonValue]> {
    obj.get(key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("snapshot is missing array field {key}"))
}

fn optional_array<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a [JsonValue]> {
    match obj.get(key) {
        None => Ok(&[]),
        Some(JsonValue::Array(values)) => Ok(values),
        Some(_) => Err(format!("snapshot field {key} must be an array")),
    }
}

fn value_to_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => {
            if value.fract() == 0.0 {
                format!("{}", *value as i64)
            } else {
                format!("{value:.6}")
            }
        }
        JsonValue::String(value) => value.clone(),
        JsonValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(value_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        // 保留嵌套指标的稳定内容，避免 dtype 分布变化在 snapshot diff 中被折叠掉。
        JsonValue::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{key}: {}", value_to_string(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// active 与 suppressed 使用独立键，确保 suppression 变化不会被同 ID 折叠。
fn all_diagnostic_severities(snapshot: &Snapshot) -> BTreeMap<DiagnosticKey, String> {
    let mut severities = BTreeMap::new();
    for (key, severity) in &snapshot.diagnostic_severities {
        severities.insert((key.clone(), DiagnosticState::Active), severity.clone());
    }
    for (key, severity) in &snapshot.suppressed_diagnostic_severities {
        severities.insert(
            (key.clone(), DiagnosticState::Suppressed),
            severity.clone(),
        );
    }
    severities
}

fn diagnostic_key_label(key: &DiagnosticKey) -> String {
    match key.1 {
        DiagnosticState::Active => key.0.clone(),
        DiagnosticState::Suppressed => format!("{} (suppressed)", key.0),
    }
}

fn diagnostic_label(
    key: &DiagnosticKey,
    severities: &BTreeMap<DiagnosticKey, String>,
) -> String {
    severities
        .get(key)
        .map(|severity| match key.1 {
            DiagnosticState::Active => format!("{} [{severity}]", key.0),
            DiagnosticState::Suppressed => {
                format!("{} [{severity}, suppressed]", key.0)
            }
        })
        .unwrap_or_else(|| diagnostic_key_label(key))
}

fn severity_rank(severity: &str) -> u8 {
    match severity {
        "error" => 2,
        "warning" => 1,
        _ => 0,
    }
}

fn list_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(", ")
    }
}

fn json_string_array(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", escape_json(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(status: &str, diagnostics: &str, suppressed_diagnostics: &str) -> Snapshot {
        parse_snapshot(&format!(
            r#"{{"schema":"edgefit.snapshot.v1","edgefit_version":"0.1.0","status":"{status}","model_path":"m","model_hash":"sha256:x","target_id":"t","target_profile_source":"test","target_profile_confidence":"seed","target_profile_last_verified":"2026-07-10","target_profile_fingerprint":"fnv1a64:test","metrics":{{"a":1}},"diagnostics":{diagnostics},"suppressed_diagnostics":{suppressed_diagnostics}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parses_snapshot_ids() {
        let snapshot = parse_snapshot(r#"{"schema":"edgefit.snapshot.v1","edgefit_version":"0.1.0","status":"fail","model_path":"m","model_hash":"sha256:x","target_id":"t","target_profile_source":"test","target_profile_confidence":"seed","target_profile_last_verified":"2026-07-10","target_profile_fingerprint":"fnv1a64:test","metrics":{"a":1},"diagnostics":[{"id":"EF1","severity":"error"}]}"#).unwrap();
        assert_eq!(snapshot.status, "fail");
        assert!(snapshot.diagnostic_ids.contains("EF1"));
    }

    #[test]
    fn parses_suppressed_diagnostics_from_legacy_report() {
        let snapshot = parse_snapshot(r#"{"schema":"edgefit.report.v1","edgefit_version":"0.1.0","status":"pass","model":{"path":"m","sha256":"sha256:x"},"target":{"id":"t","profile_metadata":{"source":"test","confidence":"seed","last_verified":"2026-07-10","fingerprint":"fnv1a64:test"}},"metrics":{"a":1},"diagnostics":[],"suppressed_diagnostics":[{"id":"EF1","severity":"error","location":"model.file"}]}"#).unwrap();
        assert_eq!(
            snapshot
                .suppressed_diagnostic_severities
                .get("EF1@model.file")
                .map(String::as_str),
            Some("error")
        );
    }

    #[test]
    fn new_suppressed_error_is_visible_and_blocking() {
        let old = snapshot("pass", "[]", "[]");
        let new = snapshot(
            "pass",
            "[]",
            r#"[{"id":"EF1","severity":"error"}]"#,
        );

        let diff = diff_snapshots(&old, &new).unwrap();
        assert_eq!(diff.status, "fail");
        assert_eq!(diff.added_diagnostics, vec!["EF1 [error, suppressed]"]);
    }

    #[test]
    fn suppressing_active_error_is_visible_and_blocking() {
        let old = snapshot(
            "fail",
            r#"[{"id":"EF1","severity":"error"}]"#,
            "[]",
        );
        let new = snapshot(
            "pass",
            "[]",
            r#"[{"id":"EF1","severity":"error"}]"#,
        );

        let diff = diff_snapshots(&old, &new).unwrap();
        assert_eq!(diff.status, "fail");
        assert_eq!(diff.added_diagnostics, vec!["EF1 [error, suppressed]"]);
        assert_eq!(diff.resolved_diagnostics, vec!["EF1 [error]"]);
    }

    #[test]
    fn unchanged_suppressed_error_does_not_block() {
        let old = snapshot(
            "pass",
            "[]",
            r#"[{"id":"EF1","severity":"error"}]"#,
        );
        let new = old.clone();

        let diff = diff_snapshots(&old, &new).unwrap();
        assert_eq!(diff.status, "pass");
        assert!(diff.added_diagnostics.is_empty());
        assert!(diff.resolved_diagnostics.is_empty());
    }

    #[test]
    fn new_suppressed_warning_is_visible_without_blocking() {
        let old = snapshot("pass", "[]", "[]");
        let new = snapshot(
            "pass",
            "[]",
            r#"[{"id":"EF1","severity":"warning"}]"#,
        );

        let diff = diff_snapshots(&old, &new).unwrap();
        assert_eq!(diff.status, "pass");
        assert_eq!(diff.added_diagnostics, vec!["EF1 [warning, suppressed]"]);
    }
}
