//! EdgeFit 报告渲染层：将统一分析结果输出为文本、Markdown、JSON、SARIF 和稳定快照。
//! 完整内存分配轨迹只进入 JSON 报告；快照保留稳定摘要，避免逐张量偏移制造差异噪声。

use edgefit_analyze::{MemoryAllocationTrace, Metrics, PeakMemoryContributor};
use edgefit_ir::{escape_json, NormalizedModel};
use edgefit_policy::{Diagnostic, PolicyResult};
use edgefit_target::TargetProfile;
use std::collections::BTreeSet;

pub const EDGEFIT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub status: String,
    pub model_path: String,
    pub model_sha256: String,
    pub target_id: String,
    pub target_source: String,
    pub target_profile_source: String,
    pub target_profile_confidence: String,
    pub target_profile_last_verified: String,
    pub target_profile_fingerprint: String,
    pub metrics: Metrics,
    pub policy: PolicyResult,
}

pub fn build_report(
    model: &NormalizedModel,
    profile: &TargetProfile,
    metrics: Metrics,
    policy: PolicyResult,
) -> Report {
    Report {
        status: policy.status.clone(),
        model_path: model.path.clone(),
        model_sha256: model.sha256.clone(),
        target_id: profile.target_id.clone(),
        target_source: profile.source.display().to_string(),
        target_profile_source: profile.metadata.source.clone(),
        target_profile_confidence: profile.metadata.confidence.clone(),
        target_profile_last_verified: profile.metadata.last_verified.clone(),
        target_profile_fingerprint: profile.fingerprint.clone(),
        metrics,
        policy,
    }
}

pub fn render_report(report: &Report, format: &str) -> String {
    match format {
        "json" => render_json(report),
        "markdown" => render_markdown(report),
        "sarif" => render_sarif(report),
        _ => render_text(report),
    }
}

pub fn render_text(report: &Report) -> String {
    let mut out = String::new();
    out.push_str(&format!("EdgeFit status: {}\n", report.status));
    out.push_str(&format!("Model: {}\n", report.model_path));
    out.push_str(&format!("Target: {}\n", report.target_id));
    out.push_str(&format!(
        "Profile confidence: {}\n\n",
        report.target_profile_confidence
    ));
    out.push_str("Metrics:\n");
    for (key, value) in metric_pairs(&report.metrics) {
        out.push_str(&format!("  {key}: {value}\n"));
    }
    out.push_str("\nDiagnostics:\n");
    if report.policy.diagnostics.is_empty() {
        out.push_str("  none\n");
    } else {
        for diag in &report.policy.diagnostics {
            out.push_str(&format!(
                "  [{}] {} {}\n",
                diag.severity, diag.id, diag.title
            ));
            out.push_str(&format!("      {}\n", diag.message));
            if let Some(location) = &diag.location {
                out.push_str(&format!("      location: {}\n", location));
            }
        }
    }
    if !report.policy.suppressed_diagnostics.is_empty() {
        out.push_str("\nSuppressed diagnostics:\n");
        for diag in &report.policy.suppressed_diagnostics {
            out.push_str(&format!(
                "  [{}] {} {}\n",
                diag.severity, diag.id, diag.title
            ));
            out.push_str(&format!("      {}\n", diag.message));
            if let Some(location) = &diag.location {
                out.push_str(&format!("      location: {}\n", location));
            }
        }
    }
    out
}

pub fn render_markdown(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("# EdgeFit Report\n\n");
    out.push_str(&format!(
        "**Status:** {}\n",
        markdown_code_span(&report.status)
    ));
    out.push_str(&format!(
        "**Model:** {}\n",
        markdown_code_span(&report.model_path)
    ));
    out.push_str(&format!(
        "**Target:** {}\n",
        markdown_code_span(&report.target_id)
    ));
    out.push_str(&format!(
        "**Profile confidence:** {}\n\n",
        markdown_code_span(&report.target_profile_confidence)
    ));
    out.push_str("## Metrics\n\n| Metric | Value |\n| --- | --- |\n");
    for (key, value) in metric_pairs(&report.metrics) {
        out.push_str(&format!(
            "| `{key}` | {} |\n",
            markdown_table_value(&value)
        ));
    }
    out.push_str("\n## Diagnostics\n\n");
    if report.policy.diagnostics.is_empty() {
        out.push_str("No diagnostics.\n");
    } else {
        out.push_str(
            "| ID | Severity | Category | Location | Message |\n| --- | --- | --- | --- | --- |\n",
        );
        for diag in &report.policy.diagnostics {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                markdown_table_value(&diag.id),
                markdown_table_value(&diag.severity),
                markdown_table_value(&diag.category),
                markdown_table_value(diag.location.as_deref().unwrap_or("")),
                markdown_table_value(&diag.message)
            ));
        }
    }
    if !report.policy.suppressed_diagnostics.is_empty() {
        out.push_str("\n## Suppressed Diagnostics\n\n");
        out.push_str(
            "| ID | Severity | Category | Location | Message |\n| --- | --- | --- | --- | --- |\n",
        );
        for diag in &report.policy.suppressed_diagnostics {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                markdown_table_value(&diag.id),
                markdown_table_value(&diag.severity),
                markdown_table_value(&diag.category),
                markdown_table_value(diag.location.as_deref().unwrap_or("")),
                markdown_table_value(&diag.message)
            ));
        }
    }
    out
}
pub fn render_json(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": \"edgefit.report.v1\",\n");
    out.push_str(&format!(
        "  \"edgefit_version\": \"{}\",\n",
        escape_json(EDGEFIT_VERSION)
    ));
    out.push_str(&format!(
        "  \"status\": \"{}\",\n",
        escape_json(&report.status)
    ));
    out.push_str("  \"model\": {\n");
    out.push_str(&format!(
        "    \"path\": \"{}\",\n",
        escape_json(&report.model_path)
    ));
    out.push_str(&format!(
        "    \"sha256\": \"{}\"\n",
        escape_json(&report.model_sha256)
    ));
    out.push_str("  },\n");
    out.push_str("  \"target\": {\n");
    out.push_str(&format!(
        "    \"id\": \"{}\",\n",
        escape_json(&report.target_id)
    ));
    out.push_str(&format!(
        "    \"path\": \"{}\",\n",
        escape_json(&report.target_source)
    ));
    out.push_str("    \"profile_metadata\": {\n");
    out.push_str(&format!(
        "      \"source\": \"{}\",\n",
        escape_json(&report.target_profile_source)
    ));
    out.push_str(&format!(
        "      \"confidence\": \"{}\",\n",
        escape_json(&report.target_profile_confidence)
    ));
    out.push_str(&format!(
        "      \"last_verified\": \"{}\",\n",
        escape_json(&report.target_profile_last_verified)
    ));
    out.push_str(&format!(
        "      \"fingerprint\": \"{}\"\n",
        escape_json(&report.target_profile_fingerprint)
    ));
    out.push_str("    }\n");
    out.push_str("  },\n");
    out.push_str("  \"metrics\": ");
    out.push_str(&render_metrics_json(&report.metrics, 2, true));
    out.push_str(",\n");
    out.push_str("  \"diagnostics\": ");
    out.push_str(&render_diagnostics_json(&report.policy.diagnostics, 2));
    out.push_str(",\n");
    out.push_str("  \"suppressed_diagnostics\": ");
    out.push_str(&render_diagnostics_json(
        &report.policy.suppressed_diagnostics,
        2,
    ));
    out.push('\n');
    out.push_str("}\n");
    out
}

/// 生成独立、稳定的 snapshot schema，避免把完整 report JSON 伪装成基线快照。
pub fn render_snapshot(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": \"edgefit.snapshot.v1\",\n");
    out.push_str(&format!(
        "  \"edgefit_version\": \"{}\",\n",
        escape_json(EDGEFIT_VERSION)
    ));
    out.push_str(&format!(
        "  \"status\": \"{}\",\n",
        escape_json(&report.status)
    ));
    out.push_str(&format!(
        "  \"model_path\": \"{}\",\n",
        escape_json(&report.model_path)
    ));
    out.push_str(&format!(
        "  \"model_hash\": \"{}\",\n",
        escape_json(&report.model_sha256)
    ));
    out.push_str(&format!(
        "  \"target_id\": \"{}\",\n",
        escape_json(&report.target_id)
    ));
    out.push_str(&format!(
        "  \"target_profile_source\": \"{}\",\n",
        escape_json(&report.target_profile_source)
    ));
    out.push_str(&format!(
        "  \"target_profile_confidence\": \"{}\",\n",
        escape_json(&report.target_profile_confidence)
    ));
    out.push_str(&format!(
        "  \"target_profile_last_verified\": \"{}\",\n",
        escape_json(&report.target_profile_last_verified)
    ));
    out.push_str(&format!(
        "  \"target_profile_fingerprint\": \"{}\",\n",
        escape_json(&report.target_profile_fingerprint)
    ));
    out.push_str("  \"metrics\": ");
    out.push_str(&render_metrics_json(&report.metrics, 2, false));
    out.push_str(",\n");
    out.push_str("  \"diagnostics\": ");
    out.push_str(&render_diagnostics_json(&report.policy.diagnostics, 2));
    out.push_str(",\n");
    out.push_str("  \"suppressed_diagnostics\": ");
    out.push_str(&render_diagnostics_json(
        &report.policy.suppressed_diagnostics,
        2,
    ));
    out.push('\n');
    out.push_str("}\n");
    out
}

fn render_diagnostics_json(diagnostics: &[Diagnostic], indent: usize) -> String {
    let pad = " ".repeat(indent);
    let inner = " ".repeat(indent + 2);
    let field = " ".repeat(indent + 4);
    let mut out = String::new();
    out.push('[');
    if !diagnostics.is_empty() {
        out.push('\n');
        for (index, diag) in diagnostics.iter().enumerate() {
            out.push_str(&format!("{inner}{{\n"));
            out.push_str(&format!("{field}\"id\": \"{}\",\n", escape_json(&diag.id)));
            out.push_str(&format!(
                "{field}\"severity\": \"{}\",\n",
                escape_json(&diag.severity)
            ));
            out.push_str(&format!(
                "{field}\"category\": \"{}\",\n",
                escape_json(&diag.category)
            ));
            if let Some(location) = &diag.location {
                out.push_str(&format!(
                    "{field}\"location\": \"{}\",\n",
                    escape_json(location)
                ));
            }
            out.push_str(&format!(
                "{field}\"title\": \"{}\",\n",
                escape_json(&diag.title)
            ));
            out.push_str(&format!(
                "{field}\"message\": \"{}\"",
                escape_json(&diag.message)
            ));
            if let Some(actual) = &diag.actual {
                out.push_str(&format!(
                    ",\n{field}\"actual\": \"{}\"",
                    escape_json(actual)
                ));
            }
            if let Some(budget) = &diag.budget {
                out.push_str(&format!(
                    ",\n{field}\"budget\": \"{}\"",
                    escape_json(budget)
                ));
            }
            if !diag.suggestions.is_empty() {
                out.push_str(&format!(",\n{field}\"suggestions\": ["));
                for (idx, suggestion) in diag.suggestions.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&format!("\"{}\"", escape_json(suggestion)));
                }
                out.push(']');
            }
            out.push_str(&format!("\n{inner}}}"));
            if index + 1 != diagnostics.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str(&pad);
    }
    out.push(']');
    out
}
pub fn render_sarif(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"$schema\": \"https://json.schemastore.org/sarif-2.1.0.json\",\n");
    out.push_str("  \"version\": \"2.1.0\",\n");
    out.push_str("  \"runs\": [\n    {\n");
    out.push_str("      \"tool\": {\n        \"driver\": {\n");
    out.push_str("          \"name\": \"edgefit\",\n");
    out.push_str(&format!(
        "          \"semanticVersion\": \"{}\",\n",
        EDGEFIT_VERSION
    ));
    let mut seen_rule_ids = BTreeSet::new();
    let rules = report
        .policy
        .diagnostics
        .iter()
        .filter(|diagnostic| seen_rule_ids.insert(diagnostic.id.as_str()))
        .collect::<Vec<_>>();
    out.push_str("          \"rules\": [");
    if !rules.is_empty() {
        out.push('\n');
        for (index, diag) in rules.iter().enumerate() {
            out.push_str(&format!(
                "            {{ \"id\": \"{}\", \"name\": \"{}\", \"shortDescription\": {{ \"text\": \"{}\" }} }}",
                escape_json(&diag.id),
                escape_json(&diag.title.to_ascii_lowercase().replace(' ', "-")),
                escape_json(&diag.title)
            ));
            if index + 1 != rules.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("          ");
    }
    out.push_str("]\n        }\n      },\n");
    out.push_str("      \"results\": [");
    if !report.policy.diagnostics.is_empty() {
        out.push('\n');
        for (index, diag) in report.policy.diagnostics.iter().enumerate() {
            out.push_str(&format!(
                "        {{ \"ruleId\": \"{}\", \"level\": \"{}\", \"message\": {{ \"text\": \"{}\" }}, \"locations\": [{}]{} }}",
                escape_json(&diag.id),
                if diag.severity == "error" { "error" } else { "warning" },
                escape_json(&diag.message),
                render_sarif_location(report, diag),
                render_sarif_fingerprints(diag)
            ));
            if index + 1 != report.policy.diagnostics.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("      ");
    }
    out.push_str("]\n    }\n  ]\n}\n");
    out
}

fn render_sarif_location(report: &Report, diag: &Diagnostic) -> String {
    let mut out = format!(
        "{{ \"physicalLocation\": {{ \"artifactLocation\": {{ \"uri\": \"{}\" }} }}",
        escape_json(&report.model_path)
    );
    if let Some(location) = &diag.location {
        out.push_str(&format!(
            ", \"logicalLocations\": [{{ \"name\": \"{}\", \"fullyQualifiedName\": \"{}:{}\" }}]",
            escape_json(location),
            escape_json(&diag.id),
            escape_json(location)
        ));
    }
    out.push_str(" }");
    out
}

fn render_sarif_fingerprints(diag: &Diagnostic) -> String {
    match &diag.location {
        Some(location) => format!(
            ", \"partialFingerprints\": {{ \"edgefitLocation\": \"{}:{}\" }}",
            escape_json(&diag.id),
            escape_json(location)
        ),
        None => String::new(),
    }
}
fn metric_pairs(metrics: &Metrics) -> Vec<(String, String)> {
    vec![
        (
            "model_file_bytes".to_string(),
            metrics.model_file_bytes.to_string(),
        ),
        (
            "external_data_file_count".to_string(),
            metrics.external_data_file_count.to_string(),
        ),
        (
            "opset_versions".to_string(),
            format_opset_versions(&metrics.opset_versions),
        ),
        (
            "initializer_bytes".to_string(),
            metrics.initializer_bytes.to_string(),
        ),
        (
            "unresolved_initializer_size_count".to_string(),
            metrics.unresolved_initializer_size_count.to_string(),
        ),
        (
            "estimated_peak_activation_bytes".to_string(),
            metrics.estimated_peak_activation_bytes.to_string(),
        ),
        (
            "peak_activation_confidence".to_string(),
            metrics.peak_activation_confidence.clone(),
        ),
        (
            "planned_activation_arena_bytes".to_string(),
            metrics.planned_activation_arena_bytes.to_string(),
        ),
        (
            "activation_tensor_alignment_bytes".to_string(),
            metrics.activation_tensor_alignment_bytes.to_string(),
        ),
        (
            "activation_planner_algorithm".to_string(),
            metrics.activation_planner_algorithm.clone(),
        ),
        (
            "activation_planning_overflowed".to_string(),
            metrics.activation_planning_overflowed.to_string(),
        ),
        (
            "peak_activation_event".to_string(),
            metrics.peak_activation_event.to_string(),
        ),
        (
            "peak_activation_node".to_string(),
            format_peak_activation_node(metrics),
        ),
        (
            "peak_activation_live_allocated_bytes".to_string(),
            metrics.peak_activation_live_allocated_bytes.to_string(),
        ),
        (
            "peak_activation_workspace_bytes".to_string(),
            metrics.peak_activation_workspace_bytes.to_string(),
        ),
        (
            "peak_activation_fragmentation_bytes".to_string(),
            metrics.peak_activation_fragmentation_bytes.to_string(),
        ),
        (
            "inplace_reuse_count".to_string(),
            metrics.inplace_reuse_count.to_string(),
        ),
        (
            "inplace_avoided_allocation_bytes".to_string(),
            metrics.inplace_avoided_allocation_bytes.to_string(),
        ),
        (
            "peak_activation_contributors".to_string(),
            format_peak_memory_contributors(&metrics.peak_activation_contributors),
        ),
        (
            "shape_inference_status".to_string(),
            metrics.shape_inference_status.clone(),
        ),
        (
            "shape_inference_error".to_string(),
            metrics
                .shape_inference_error
                .clone()
                .unwrap_or_else(|| "none".to_string()),
        ),
        (
            "dynamic_tensor_count".to_string(),
            metrics.dynamic_tensor_count.to_string(),
        ),
        (
            "bounded_dynamic_tensor_count".to_string(),
            metrics.bounded_dynamic_tensor_count.to_string(),
        ),
        (
            "unresolved_tensor_size_count".to_string(),
            metrics.unresolved_tensor_size_count.to_string(),
        ),
        (
            "unsupported_ops".to_string(),
            format_string_array(&metrics.unsupported_ops),
        ),
        (
            "unsupported_dtypes".to_string(),
            format_string_array(&metrics.unsupported_dtypes),
        ),
        (
            "unknown_dtype_tensors".to_string(),
            format_string_array(&metrics.unknown_dtype_tensors),
        ),
        (
            "tensor_rank_violations".to_string(),
            format_tensor_rank_violations(&metrics.tensor_rank_violations),
        ),
        (
            "op_dtype_violations".to_string(),
            format_op_dtype_violations(&metrics.op_dtype_violations),
        ),
        (
            "op_rank_violations".to_string(),
            format_op_rank_violations(&metrics.op_rank_violations),
        ),
        (
            "quantized_initializer_fraction".to_string(),
            format!("{:.6}", metrics.quantized_initializer_fraction),
        ),
        (
            "initializer_dtype_distribution".to_string(),
            format_initializer_dtype_distribution(&metrics.initializer_dtype_distribution),
        ),
        (
            "quantization_representation".to_string(),
            metrics.quantization_representation.clone(),
        ),
        (
            "qdq_covered_node_count".to_string(),
            metrics.qdq_covered_node_count.to_string(),
        ),
        (
            "qoperator_node_count".to_string(),
            metrics.qoperator_node_count.to_string(),
        ),
        (
            "quantization_eligible_node_count".to_string(),
            metrics.quantization_eligible_node_count.to_string(),
        ),
        (
            "quantization_covered_node_count".to_string(),
            metrics.quantization_covered_node_count.to_string(),
        ),
        (
            "quantization_operator_coverage".to_string(),
            format!("{:.6}", metrics.quantization_operator_coverage),
        ),
        (
            "target_requires_int8".to_string(),
            metrics.target_requires_int8.to_string(),
        ),
        (
            "int8_tensor_count".to_string(),
            metrics.int8_tensor_count.to_string(),
        ),
        (
            "uint8_tensor_count".to_string(),
            metrics.uint8_tensor_count.to_string(),
        ),
        (
            "non_int8_graph_io".to_string(),
            format_graph_io_dtype_violations(&metrics.non_int8_graph_io),
        ),
        (
            "ops_without_int8_path".to_string(),
            format_string_array(&metrics.ops_without_int8_path),
        ),
        ("node_count".to_string(), metrics.node_count.to_string()),
        ("tensor_count".to_string(), metrics.tensor_count.to_string()),
    ]
}

fn render_metrics_json(metrics: &Metrics, indent: usize, include_trace: bool) -> String {
    let pad = " ".repeat(indent);
    let inner = " ".repeat(indent + 2);
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "{inner}\"model_file_bytes\": {},\n",
        metrics.model_file_bytes
    ));
    out.push_str(&format!(
        "{inner}\"external_data_file_count\": {},\n",
        metrics.external_data_file_count
    ));
    out.push_str(&format!(
        "{inner}\"opset_versions\": {},\n",
        opset_versions_json(&metrics.opset_versions)
    ));
    out.push_str(&format!(
        "{inner}\"initializer_bytes\": {},\n",
        metrics.initializer_bytes
    ));
    out.push_str(&format!(
        "{inner}\"unresolved_initializer_size_count\": {},\n",
        metrics.unresolved_initializer_size_count
    ));
    out.push_str(&format!(
        "{inner}\"estimated_peak_activation_bytes\": {},\n",
        metrics.estimated_peak_activation_bytes
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_confidence\": \"{}\",\n",
        escape_json(&metrics.peak_activation_confidence)
    ));
    out.push_str(&format!(
        "{inner}\"planned_activation_arena_bytes\": {},\n",
        metrics.planned_activation_arena_bytes
    ));
    out.push_str(&format!(
        "{inner}\"activation_tensor_alignment_bytes\": {},\n",
        metrics.activation_tensor_alignment_bytes
    ));
    out.push_str(&format!(
        "{inner}\"activation_planner_algorithm\": \"{}\",\n",
        escape_json(&metrics.activation_planner_algorithm)
    ));
    out.push_str(&format!(
        "{inner}\"activation_planning_overflowed\": {},\n",
        metrics.activation_planning_overflowed
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_event\": \"{}\",\n",
        escape_json(&metrics.peak_activation_event)
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_node_index\": {},\n",
        optional_u64_json(metrics.peak_activation_node_index)
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_node_name\": {},\n",
        optional_string_json(metrics.peak_activation_node_name.as_deref())
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_op_type\": {},\n",
        optional_string_json(metrics.peak_activation_op_type.as_deref())
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_live_allocated_bytes\": {},\n",
        metrics.peak_activation_live_allocated_bytes
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_workspace_bytes\": {},\n",
        metrics.peak_activation_workspace_bytes
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_fragmentation_bytes\": {},\n",
        metrics.peak_activation_fragmentation_bytes
    ));
    out.push_str(&format!(
        "{inner}\"inplace_reuse_count\": {},\n",
        metrics.inplace_reuse_count
    ));
    out.push_str(&format!(
        "{inner}\"inplace_avoided_allocation_bytes\": {},\n",
        metrics.inplace_avoided_allocation_bytes
    ));
    out.push_str(&format!(
        "{inner}\"peak_activation_contributors\": {},\n",
        peak_memory_contributors_json(&metrics.peak_activation_contributors)
    ));
    if include_trace {
        out.push_str(&format!(
            "{inner}\"activation_allocation_trace\": {},\n",
            memory_allocation_trace_json(&metrics.activation_allocation_trace)
        ));
    }
    out.push_str(&format!(
        "{inner}\"shape_inference_status\": \"{}\",\n",
        escape_json(&metrics.shape_inference_status)
    ));
    match &metrics.shape_inference_error {
        Some(error) => out.push_str(&format!(
            "{inner}\"shape_inference_error\": \"{}\",\n",
            escape_json(error)
        )),
        None => out.push_str(&format!("{inner}\"shape_inference_error\": null,\n")),
    }
    out.push_str(&format!(
        "{inner}\"dynamic_tensor_count\": {},\n",
        metrics.dynamic_tensor_count
    ));
    out.push_str(&format!(
        "{inner}\"bounded_dynamic_tensor_count\": {},\n",
        metrics.bounded_dynamic_tensor_count
    ));
    out.push_str(&format!(
        "{inner}\"unresolved_tensor_size_count\": {},\n",
        metrics.unresolved_tensor_size_count
    ));
    out.push_str(&format!(
        "{inner}\"unsupported_ops\": {},\n",
        json_string_array(&metrics.unsupported_ops)
    ));
    out.push_str(&format!(
        "{inner}\"unsupported_dtypes\": {},\n",
        json_string_array(&metrics.unsupported_dtypes)
    ));
    out.push_str(&format!(
        "{inner}\"unknown_dtype_tensors\": {},\n",
        json_string_array(&metrics.unknown_dtype_tensors)
    ));
    out.push_str(&format!(
        "{inner}\"tensor_rank_violations\": {},\n",
        json_string_array(&tensor_rank_violation_strings(
            &metrics.tensor_rank_violations
        ))
    ));
    out.push_str(&format!(
        "{inner}\"op_dtype_violations\": {},\n",
        json_string_array(&op_dtype_violation_strings(&metrics.op_dtype_violations))
    ));
    out.push_str(&format!(
        "{inner}\"op_rank_violations\": {},\n",
        json_string_array(&op_rank_violation_strings(&metrics.op_rank_violations))
    ));
    out.push_str(&format!(
        "{inner}\"quantized_initializer_fraction\": {:.6},\n",
        metrics.quantized_initializer_fraction
    ));
    out.push_str(&format!(
        "{inner}\"initializer_dtype_distribution\": {},\n",
        initializer_dtype_distribution_json(&metrics.initializer_dtype_distribution)
    ));
    out.push_str(&format!(
        "{inner}\"quantization_representation\": \"{}\",\n",
        escape_json(&metrics.quantization_representation)
    ));
    out.push_str(&format!(
        "{inner}\"qdq_covered_node_count\": {},\n",
        metrics.qdq_covered_node_count
    ));
    out.push_str(&format!(
        "{inner}\"qoperator_node_count\": {},\n",
        metrics.qoperator_node_count
    ));
    out.push_str(&format!(
        "{inner}\"quantization_eligible_node_count\": {},\n",
        metrics.quantization_eligible_node_count
    ));
    out.push_str(&format!(
        "{inner}\"quantization_covered_node_count\": {},\n",
        metrics.quantization_covered_node_count
    ));
    out.push_str(&format!(
        "{inner}\"quantization_operator_coverage\": {:.6},\n",
        metrics.quantization_operator_coverage
    ));
    out.push_str(&format!(
        "{inner}\"target_requires_int8\": {},\n",
        metrics.target_requires_int8
    ));
    out.push_str(&format!(
        "{inner}\"int8_tensor_count\": {},\n",
        metrics.int8_tensor_count
    ));
    out.push_str(&format!(
        "{inner}\"uint8_tensor_count\": {},\n",
        metrics.uint8_tensor_count
    ));
    out.push_str(&format!(
        "{inner}\"non_int8_graph_io\": {},\n",
        json_string_array(&graph_io_dtype_violation_strings(
            &metrics.non_int8_graph_io
        ))
    ));
    out.push_str(&format!(
        "{inner}\"ops_without_int8_path\": {},\n",
        json_string_array(&metrics.ops_without_int8_path)
    ));
    out.push_str(&format!("{inner}\"node_count\": {},\n", metrics.node_count));
    out.push_str(&format!(
        "{inner}\"tensor_count\": {}\n",
        metrics.tensor_count
    ));
    out.push_str(&format!("{pad}}}"));
    out
}

fn format_peak_activation_node(metrics: &Metrics) -> String {
    let mut parts = Vec::new();
    if let Some(index) = metrics.peak_activation_node_index {
        parts.push(format!("#{index}"));
    }
    if let Some(name) = metrics.peak_activation_node_name.as_deref() {
        parts.push(name.to_string());
    }
    if let Some(op_type) = metrics.peak_activation_op_type.as_deref() {
        parts.push(format!("({op_type})"));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" ")
    }
}

fn format_peak_memory_contributors(values: &[PeakMemoryContributor]) -> String {
    let values = values
        .iter()
        .map(|item| format!("{}:{} bytes", item.tensor, item.allocated_bytes))
        .collect::<Vec<_>>();
    format_string_array(&values)
}

fn peak_memory_contributors_json(values: &[PeakMemoryContributor]) -> String {
    let items = values
        .iter()
        .map(|item| {
            format!(
                "{{\"tensor\":\"{}\",\"allocated_bytes\":{}}}",
                escape_json(&item.tensor),
                item.allocated_bytes
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn memory_allocation_trace_json(values: &[MemoryAllocationTrace]) -> String {
    let items = values
        .iter()
        .map(|item| {
            format!(
                concat!(
                    "{{\"kind\":\"{}\",\"name\":\"{}\",",
                    "\"logical_bytes\":{},\"allocated_bytes\":{},",
                    "\"arena_offset\":{},\"first_event\":{},\"last_event\":{},",
                    "\"alias_of\":{},\"size_source\":\"{}\",\"graph_output\":{}}}"
                ),
                escape_json(&item.kind),
                escape_json(&item.name),
                optional_u64_json(item.logical_bytes),
                optional_u64_json(item.allocated_bytes),
                optional_u64_json(item.arena_offset),
                item.first_event,
                item.last_event,
                optional_string_json(item.alias_of.as_deref()),
                escape_json(&item.size_source),
                item.graph_output
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn optional_u64_json(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn optional_string_json(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", escape_json(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn tensor_rank_violation_strings(values: &[edgefit_analyze::TensorRankViolation]) -> Vec<String> {
    values
        .iter()
        .map(|item| format!("{}:{}>{}", item.tensor, item.rank, item.max_rank))
        .collect()
}

fn op_dtype_violation_strings(values: &[edgefit_analyze::OpDtypeViolation]) -> Vec<String> {
    values
        .iter()
        .map(|item| format!("{}:{}:{}", item.op_type, item.tensor, item.dtype))
        .collect()
}

fn op_rank_violation_strings(values: &[edgefit_analyze::OpRankViolation]) -> Vec<String> {
    values
        .iter()
        .map(|item| {
            format!(
                "{}:{}:{}>{}",
                item.op_type, item.tensor, item.rank, item.max_rank
            )
        })
        .collect()
}

fn graph_io_dtype_violation_strings(
    values: &[edgefit_analyze::GraphIoDtypeViolation],
) -> Vec<String> {
    values
        .iter()
        .map(|item| format!("{}:{}:{}", item.boundary, item.tensor, item.dtype))
        .collect()
}

fn format_tensor_rank_violations(values: &[edgefit_analyze::TensorRankViolation]) -> String {
    format_string_array(&tensor_rank_violation_strings(values))
}

fn format_op_dtype_violations(values: &[edgefit_analyze::OpDtypeViolation]) -> String {
    format_string_array(&op_dtype_violation_strings(values))
}

fn format_op_rank_violations(values: &[edgefit_analyze::OpRankViolation]) -> String {
    format_string_array(&op_rank_violation_strings(values))
}

fn format_graph_io_dtype_violations(
    values: &[edgefit_analyze::GraphIoDtypeViolation],
) -> String {
    format_string_array(&graph_io_dtype_violation_strings(values))
}

fn format_initializer_dtype_distribution(
    values: &[edgefit_analyze::InitializerDtypeMetric],
) -> String {
    let items = values
        .iter()
        .map(|item| format!("{}:{} tensors/{} bytes", item.dtype, item.tensor_count, item.bytes))
        .collect::<Vec<_>>();
    format_string_array(&items)
}

fn initializer_dtype_distribution_json(
    values: &[edgefit_analyze::InitializerDtypeMetric],
) -> String {
    let items = values
        .iter()
        .map(|item| {
            format!(
                "{{\"dtype\":\"{}\",\"tensor_count\":{},\"bytes\":{}}}",
                escape_json(&item.dtype),
                item.tensor_count,
                item.bytes
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn format_opset_versions(values: &std::collections::BTreeMap<String, u64>) -> String {
    if values.is_empty() {
        "{}".to_string()
    } else {
        format!(
            "{{{}}}",
            values
                .iter()
                .map(|(domain, version)| format!("{domain}: {version}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn opset_versions_json(values: &std::collections::BTreeMap<String, u64>) -> String {
    let items = values
        .iter()
        .map(|(domain, version)| format!("\"{}\": {version}", escape_json(domain)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{items}}}")
}

fn json_string_array(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| format!("\"{}\"", escape_json(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn format_string_array(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", values.join(", "))
    }
}

/// 使用比内容中连续反引号更长的围栏，避免模型路径或 profile 文本截断 Markdown 行内代码。
fn markdown_code_span(value: &str) -> String {
    let value = value
        .replace("\r\n", " ")
        .replace('\r', " ")
        .replace('\n', " ");
    let mut current_run = 0_usize;
    let mut longest_run = 0_usize;
    for ch in value.chars() {
        if ch == '`' {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    let fence = "`".repeat(longest_run + 1);
    format!("{fence}{value}{fence}")
}

/// 转义 Markdown 表格中的结构字符；换行保留为单元格内的显式换行标记。
fn markdown_table_value(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push_str("<br>");
            }
            '\n' => out.push_str("<br>"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '|' => out.push_str("&#124;"),
            '`' => out.push_str("&#96;"),
            value => out.push(value),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefit_analyze::Metrics;
    use edgefit_policy::{Diagnostic, PolicyResult};

    #[test]
    fn json_report_contains_schema_and_profile_metadata() {
        let mut report = Report {
            status: "pass".to_string(),
            model_path: "m".to_string(),
            model_sha256: "s".to_string(),
            target_id: "t".to_string(),
            target_source: "target.yaml".to_string(),
            target_profile_source: "test profile".to_string(),
            target_profile_confidence: "seed".to_string(),
            target_profile_last_verified: "2026-07-09".to_string(),
            target_profile_fingerprint: "fnv1a64:test".to_string(),
            metrics: Metrics {
                model_file_bytes: 1,
                external_data_file_count: 0,
                opset_versions: std::collections::BTreeMap::new(),
                initializer_bytes: 0,
                unresolved_initializer_size_count: 0,
                estimated_peak_activation_bytes: 0,
                peak_activation_confidence: "high".to_string(),
                planned_activation_arena_bytes: 128,
                activation_tensor_alignment_bytes: 16,
                activation_planner_algorithm: "linear_scan_best_fit_v2".to_string(),
                activation_planning_overflowed: false,
                peak_activation_event: "node".to_string(),
                peak_activation_node_index: Some(0),
                peak_activation_node_name: Some("node`\n|name".to_string()),
                peak_activation_op_type: Some("Relu".to_string()),
                peak_activation_live_allocated_bytes: 64,
                peak_activation_workspace_bytes: 32,
                peak_activation_fragmentation_bytes: 32,
                inplace_reuse_count: 1,
                inplace_avoided_allocation_bytes: 64,
                peak_activation_contributors: vec![PeakMemoryContributor {
                    tensor: "tensor`\n|name".to_string(),
                    allocated_bytes: 64,
                }],
                activation_allocation_trace: vec![MemoryAllocationTrace {
                    kind: "tensor".to_string(),
                    name: "tensor`\n|name".to_string(),
                    logical_bytes: Some(60),
                    allocated_bytes: Some(64),
                    arena_offset: Some(0),
                    first_event: 0,
                    last_event: 1,
                    alias_of: None,
                    size_source: "shape".to_string(),
                    graph_output: false,
                }],
                shape_inference_status: "not_recorded".to_string(),
                shape_inference_error: None,
                dynamic_tensor_count: 0,
                bounded_dynamic_tensor_count: 0,
                unresolved_tensor_size_count: 0,
                unsupported_ops: vec![],
                unsupported_dtypes: vec![],
                unknown_dtype_tensors: vec![],
                tensor_rank_violations: vec![],
                op_dtype_violations: vec![],
                op_rank_violations: vec![],
                quantized_initializer_fraction: 1.0,
                initializer_dtype_distribution: vec![],
                quantization_representation: "none".to_string(),
                qdq_covered_node_count: 0,
                qoperator_node_count: 0,
                quantization_eligible_node_count: 0,
                quantization_covered_node_count: 0,
                quantization_operator_coverage: 0.0,
                target_requires_int8: false,
                int8_tensor_count: 0,
                uint8_tensor_count: 0,
                non_int8_graph_io: vec![],
                ops_without_int8_path: vec![],
                node_count: 0,
                tensor_count: 0,
            },
            policy: PolicyResult {
                status: "pass".to_string(),
                diagnostics: vec![],
                suppressed_diagnostics: vec![],
            },
        };
        let json = render_json(&report);
        assert!(json.contains("edgefit.report.v1"));
        assert!(json.contains("profile_metadata"));
        assert!(json.contains("seed"));
        assert!(json.contains("\"planned_activation_arena_bytes\": 128"));
        assert!(json.contains("\"activation_planning_overflowed\": false"));
        assert!(json.contains("\"peak_activation_contributors\""));
        assert!(json.contains("\"activation_allocation_trace\""));
        assert!(json.contains("tensor`\\n|name"));

        let text = render_text(&report);
        assert!(text.contains("planned_activation_arena_bytes: 128"));
        assert!(!text.contains("activation_allocation_trace"));

        let markdown = render_markdown(&report);
        assert!(markdown.contains("tensor&#96;<br>&#124;name:64 bytes"));
        assert!(!markdown.contains("activation_allocation_trace"));

        report.policy.suppressed_diagnostics.push(Diagnostic {
            id: "EF0401".to_string(),
            severity: "error".to_string(),
            category: "memory".to_string(),
            location: Some("model.file".to_string()),
            title: "suppressed test".to_string(),
            message: "suppressed test".to_string(),
            actual: None,
            budget: None,
            suggestions: vec![],
        });
        let snapshot = render_snapshot(&report);
        assert!(snapshot.contains("\"suppressed_diagnostics\""));
        assert!(snapshot.contains("\"id\": \"EF0401\""));
        assert!(snapshot.contains("\"planned_activation_arena_bytes\": 128"));
        assert!(snapshot.contains("\"peak_activation_contributors\""));
        assert!(!snapshot.contains("\"activation_allocation_trace\""));
    }

    #[test]
    fn markdown_helpers_keep_structural_characters_inside_one_field() {
        assert_eq!(markdown_code_span("a`b\nc"), "``a`b c``");
        assert_eq!(
            markdown_table_value("a`\r\n|b"),
            "a&#96;<br>&#124;b"
        );
    }
}
