use std::fmt::Write;

use crate::{Evidence, Verification, EVIDENCE_SCHEMA, VERIFICATION_SCHEMA};

pub fn render_evidence_json(evidence: &Evidence) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"schema\": \"");
    out.push_str(EVIDENCE_SCHEMA);
    out.push_str("\",\n  \"identity\": {");
    json_string_field(&mut out, "target_id", &evidence.identity.target_id, true);
    json_string_field(&mut out, "device_id", &evidence.identity.device_id, true);
    json_string_field(
        &mut out,
        "runtime_name",
        &evidence.identity.runtime_name,
        true,
    );
    json_string_field(
        &mut out,
        "runtime_version",
        &evidence.identity.runtime_version,
        false,
    );
    out.push_str("\n  },\n  \"environment\": {");
    json_string_field(
        &mut out,
        "operating_system",
        &evidence.environment.operating_system,
        true,
    );
    json_string_field(
        &mut out,
        "architecture",
        &evidence.environment.architecture,
        true,
    );
    json_string_field(&mut out, "hardware", &evidence.environment.hardware, true);
    json_string_field(
        &mut out,
        "toolchain",
        &evidence.environment.toolchain,
        false,
    );
    out.push_str("\n  },\n  \"capture\": {");
    json_string_field(
        &mut out,
        "captured_at",
        &evidence.capture.captured_at,
        true,
    );
    json_string_field(&mut out, "command", &evidence.capture.command, true);
    json_decimal_field(&mut out, "warmup_runs", evidence.capture.warmup_runs, true);
    json_decimal_field(
        &mut out,
        "measured_runs",
        evidence.capture.measured_runs,
        false,
    );
    out.push_str("\n  },\n  \"bindings\": {");
    json_string_field(
        &mut out,
        "model_sha256",
        &evidence.bindings.model_sha256,
        true,
    );
    json_string_field(
        &mut out,
        "target_profile_sha256",
        &evidence.bindings.target_profile_sha256,
        true,
    );
    json_string_field(
        &mut out,
        "runtime_binary_sha256",
        &evidence.bindings.runtime_binary_sha256,
        false,
    );
    out.push_str("\n  },\n  \"runtime\": {\n    \"accepted\": ");
    out.push_str(if evidence.runtime.accepted {
        "true"
    } else {
        "false"
    });
    out.push_str(",\n    \"rejected_reason\": ");
    match &evidence.runtime.rejected_reason {
        Some(value) => quoted(&mut out, value),
        None => out.push_str("null"),
    }
    out.push_str(
        "\n  },\n  \"measurements\": {\n    \"arena_high_water\": {\n      \"unit\": \"bytes\",\n      \"value\": \"",
    );
    write!(out, "{}", evidence.measurements.arena_high_water_bytes).unwrap();
    out.push_str(
        "\"\n    },\n    \"latency\": {\n      \"unit\": \"ns\",\n      \"samples\": [",
    );
    for (index, sample) in evidence.measurements.latency_ns.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        write!(out, "\"{sample}\"").unwrap();
    }
    out.push_str(
        "]\n    }\n  },\n  \"thresholds\": {\n    \"arena_budget\": {\n      \"unit\": \"bytes\",\n      \"value\": \"",
    );
    write!(out, "{}", evidence.thresholds.arena_budget_bytes).unwrap();
    out.push_str(
        "\"\n    },\n    \"p95_latency_budget\": {\n      \"unit\": \"ns\",\n      \"value\": \"",
    );
    write!(out, "{}", evidence.thresholds.p95_latency_budget_ns).unwrap();
    out.push_str("\"\n    }\n  },\n  \"attachments\": [");
    for (index, attachment) in evidence.attachments.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str("\n    {");
        json_string_field(&mut out, "name", &attachment.name, true);
        json_string_field(&mut out, "path", &attachment.path, true);
        json_string_field(&mut out, "media_type", &attachment.media_type, true);
        json_decimal_field(&mut out, "bytes", attachment.bytes, true);
        json_string_field(&mut out, "sha256", &attachment.sha256, false);
        out.push_str("\n    }");
    }
    if !evidence.attachments.is_empty() {
        out.push_str("\n  ");
    }
    out.push_str("],\n  \"attestation\": {\n    \"kind\": \"none\"\n  }\n}\n");
    out
}

pub fn render_verification_json(verification: &Verification) -> String {
    let mut out = render_verification_payload(verification);
    let close = out
        .strip_suffix("\n}\n")
        .expect("internal verification JSON object");
    let mut rendered = String::with_capacity(out.len() + 100);
    rendered.push_str(close);
    rendered.push_str(",\n  \"verification_sha256\": \"");
    rendered.push_str(&escape_json(&verification.verification_sha256));
    rendered.push_str("\"\n}\n");
    out.clear();
    rendered
}

pub(crate) fn render_verification_payload(verification: &Verification) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"schema\": \"");
    out.push_str(VERIFICATION_SCHEMA);
    out.push_str("\",\n  \"status\": \"");
    out.push_str(verification.status.as_str());
    out.push_str(
        "\",\n  \"trust\": {\n    \"binding\": \"sha256-only\",\n    \"attestation\": \"none\",\n    \"profile_mutation_authority\": false,\n    \"optimizer_decision_authority\": false\n  },\n  \"evidence_sha256\": \"",
    );
    out.push_str(&escape_json(&verification.evidence_sha256));
    out.push_str("\",\n  \"expected_bindings\": {");
    json_string_field(
        &mut out,
        "model_sha256",
        &verification.expected_bindings.model_sha256,
        true,
    );
    json_string_field(
        &mut out,
        "target_profile_sha256",
        &verification.expected_bindings.target_profile_sha256,
        true,
    );
    json_string_field(
        &mut out,
        "runtime_binary_sha256",
        &verification.expected_bindings.runtime_binary_sha256,
        false,
    );
    out.push_str("\n  },\n  \"budget\": {");
    json_decimal_field(
        &mut out,
        "arena_bytes",
        verification.budget.arena_bytes,
        true,
    );
    json_decimal_field(
        &mut out,
        "p95_latency_ns",
        verification.budget.p95_latency_ns,
        false,
    );
    out.push_str("\n  },\n  \"metrics\": {");
    json_decimal_field(
        &mut out,
        "sample_count",
        verification.metrics.sample_count,
        true,
    );
    json_decimal_field(
        &mut out,
        "latency_p50_ns",
        verification.metrics.latency_p50_ns,
        true,
    );
    json_decimal_field(
        &mut out,
        "latency_p95_ns",
        verification.metrics.latency_p95_ns,
        true,
    );
    json_decimal_field(
        &mut out,
        "latency_mean_ns",
        verification.metrics.latency_mean_ns,
        true,
    );
    json_decimal_field(
        &mut out,
        "arena_utilization_ppm",
        verification.metrics.arena_utilization_ppm,
        true,
    );
    json_i128_field(
        &mut out,
        "arena_error_bytes",
        verification.metrics.arena_error_bytes,
        true,
    );
    json_i128_field(
        &mut out,
        "p95_latency_error_ns",
        verification.metrics.p95_latency_error_ns,
        false,
    );
    out.push_str("\n  },\n  \"checks\": [");
    for (index, check) in verification.checks.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str("\n    {\n      \"id\": \"");
        out.push_str(&escape_json(check.id));
        out.push_str("\",\n      \"status\": \"");
        out.push_str(check.status.as_str());
        out.push_str("\",\n      \"detail\": \"");
        out.push_str(&escape_json(&check.detail));
        out.push_str("\"\n    }");
    }
    if !verification.checks.is_empty() {
        out.push_str("\n  ");
    }
    out.push_str("]\n}\n");
    out
}

pub fn render_verification_markdown(verification: &Verification) -> String {
    let mut out = String::new();
    out.push_str("# EdgeFit calibration verification\n\n");
    writeln!(out, "- Status: **{}**", verification.status.as_str()).unwrap();
    out.push_str("- Binding: `sha256-only`\n- Attestation: `none`\n");
    out.push_str(
        "- Profile mutation authority: `false`\n- Optimizer decision authority: `false`\n",
    );
    writeln!(out, "- Evidence SHA-256: `{}`", verification.evidence_sha256).unwrap();
    writeln!(
        out,
        "- Verification SHA-256: `{}`\n",
        verification.verification_sha256
    )
    .unwrap();
    out.push_str("## Metrics\n\n| Metric | Value |\n|---|---:|\n");
    writeln!(out, "| Samples | {} |", verification.metrics.sample_count).unwrap();
    writeln!(
        out,
        "| p50 latency (ns) | {} |",
        verification.metrics.latency_p50_ns
    )
    .unwrap();
    writeln!(
        out,
        "| p95 latency (ns) | {} |",
        verification.metrics.latency_p95_ns
    )
    .unwrap();
    writeln!(
        out,
        "| Mean latency (ns) | {} |",
        verification.metrics.latency_mean_ns
    )
    .unwrap();
    writeln!(
        out,
        "| Arena utilization (ppm) | {} |",
        verification.metrics.arena_utilization_ppm
    )
    .unwrap();
    writeln!(
        out,
        "| Arena error (bytes) | {} |",
        verification.metrics.arena_error_bytes
    )
    .unwrap();
    writeln!(
        out,
        "| p95 latency error (ns) | {} |\n",
        verification.metrics.p95_latency_error_ns
    )
    .unwrap();
    out.push_str("## Checks\n\n| Check | Status | Detail |\n|---|---|---|\n");
    for check in &verification.checks {
        writeln!(
            out,
            "| `{}` | {} | {} |",
            markdown_cell(check.id),
            check.status.as_str(),
            markdown_cell(&check.detail)
        )
        .unwrap();
    }
    out
}

fn json_string_field(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("\n    \"");
    out.push_str(key);
    out.push_str("\": ");
    quoted(out, value);
    if comma {
        out.push(',');
    }
}

fn json_decimal_field(out: &mut String, key: &str, value: u64, comma: bool) {
    json_string_field(out, key, &value.to_string(), comma);
}

fn json_i128_field(out: &mut String, key: &str, value: i128, comma: bool) {
    json_string_field(out, key, &value.to_string(), comma);
}

fn quoted(out: &mut String, value: &str) {
    out.push('"');
    out.push_str(&escape_json(value));
    out.push('"');
}

fn escape_json(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value <= '\u{1f}' => write!(out, "\\u{:04x}", value as u32).unwrap(),
            value => out.push(value),
        }
    }
    out
}

fn markdown_cell(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\r', " ")
        .replace('\n', " ")
}
