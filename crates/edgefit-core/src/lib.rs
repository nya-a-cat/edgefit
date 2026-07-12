use edgefit_analyze::analyze;
use edgefit_ir::{
    load_cli_adapter_output, load_normalized_model, parse_cli_adapter_output,
    parse_normalized_model, EdgeFitResult, NormalizedModel,
};
use edgefit_policy::{evaluate, suppress_diagnostics};
use edgefit_optimize::{optimize, render_plan, OptimizationPlan};
use edgefit_report::{build_report, render_report, Report};
use edgefit_target::{load_profile, parse_profile, TargetProfile};
use std::path::{Path, PathBuf};

pub fn check_model(
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
) -> EdgeFitResult<Report> {
    check_model_with_suppressions(model_path, target_path, &[])
}

pub fn check_model_with_suppressions(
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let model = load_normalized_model(model_path)?;
    check_loaded_model(model, target_path, suppressed_ids)
}

/// 检查 CLI 从原始 ONNX 临时生成的规范化结果；外部 JSON 不得调用此入口。
pub fn check_adapter_generated_model_with_suppressions(
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let model = load_cli_adapter_output(model_path)?;
    check_loaded_model(model, target_path, suppressed_ids)
}

/// 检查内存中的规范化 JSON 与 target profile；该入口不授予 ONNX 适配来源。
pub fn check_normalized_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let model = parse_normalized_model(model_text)?;
    check_loaded_model_with_profile(
        model,
        parse_target_text(target_text, target_source)?,
        suppressed_ids,
    )
}

/// 检查受控 ONNX 适配器刚生成的内存 JSON；仅供 CLI/Python 框架的直接 ONNX 路径调用。
pub fn check_adapter_generated_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let model = parse_cli_adapter_output(model_text)?;
    check_loaded_model_with_profile(
        model,
        parse_target_text(target_text, target_source)?,
        suppressed_ids,
    )
}

/// 返回 Rust 核心生成的 canonical 单模型报告，供 Python 与其他绑定复用。
pub fn render_normalized_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    suppressed_ids: &[String],
    format: &str,
) -> EdgeFitResult<String> {
    check_normalized_text(model_text, target_text, target_source, suppressed_ids)
        .map(|report| render_report(&report, format))
}

/// 返回直接 ONNX 适配路径的 canonical 单模型报告。
pub fn render_adapter_generated_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    suppressed_ids: &[String],
    format: &str,
) -> EdgeFitResult<String> {
    check_adapter_generated_text(model_text, target_text, target_source, suppressed_ids)
        .map(|report| render_report(&report, format))
}

pub fn validate_target_text(target_text: &str, target_source: &str) -> EdgeFitResult<String> {
    parse_target_text(target_text, target_source).map(|profile| profile.target_id)
}

/// 为规范化模型生成确定性的硬件执行计划，不改写原模型。
pub fn optimize_model(
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
) -> EdgeFitResult<OptimizationPlan> {
    optimize(&load_normalized_model(model_path)?, &load_profile(target_path)?)
}

/// 为 CLI 从 ONNX 生成的可信规范化结果生成硬件执行计划。
pub fn optimize_adapter_generated_model(
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
) -> EdgeFitResult<OptimizationPlan> {
    optimize(&load_cli_adapter_output(model_path)?, &load_profile(target_path)?)
}

/// 为内存中的规范化模型与 target profile 生成 typed canonical 优化计划。
pub fn optimize_normalized_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
) -> EdgeFitResult<OptimizationPlan> {
    optimize_text(model_text, target_text, target_source, false)
}

/// 为受控 ONNX 适配器生成的内存模型与 target profile 生成 typed canonical 优化计划。
pub fn optimize_adapter_generated_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
) -> EdgeFitResult<OptimizationPlan> {
    optimize_text(model_text, target_text, target_source, true)
}

/// 为 Python 框架中的内存模型生成 canonical 优化计划。
pub fn render_optimization_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    format: &str,
    adapter_generated: bool,
) -> EdgeFitResult<String> {
    if !matches!(format, "json" | "markdown") {
        return Err("optimization format must be json or markdown".to_string());
    }
    let plan = optimize_text(model_text, target_text, target_source, adapter_generated)?;
    Ok(render_plan(&plan, format))
}

fn optimize_text(
    model_text: &str,
    target_text: &str,
    target_source: &str,
    adapter_generated: bool,
) -> EdgeFitResult<OptimizationPlan> {
    let model = if adapter_generated {
        parse_cli_adapter_output(model_text)?
    } else {
        parse_normalized_model(model_text)?
    };
    optimize(&model, &parse_target_text(target_text, target_source)?)
}

fn check_loaded_model(
    model: NormalizedModel,
    target_path: impl AsRef<Path>,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let profile = load_profile(target_path)?;
    check_loaded_model_with_profile(model, profile, suppressed_ids)
}

fn parse_target_text(text: &str, source: &str) -> EdgeFitResult<TargetProfile> {
    let profile = parse_profile(text, PathBuf::from(source))?;
    profile.validate()?;
    Ok(profile)
}

fn check_loaded_model_with_profile(
    model: NormalizedModel,
    profile: TargetProfile,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let metrics = analyze(&model, &profile);
    let policy = suppress_diagnostics(evaluate(&model, &profile, &metrics), suppressed_ids);
    Ok(build_report(&model, &profile, metrics, policy))
}
