use edgefit_analyze::analyze;
use edgefit_ir::{
    load_cli_adapter_output, load_normalized_model, EdgeFitResult, NormalizedModel,
};
use edgefit_policy::{evaluate, suppress_diagnostics};
use edgefit_report::{build_report, Report};
use edgefit_target::load_profile;
use std::path::Path;

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

fn check_loaded_model(
    model: NormalizedModel,
    target_path: impl AsRef<Path>,
    suppressed_ids: &[String],
) -> EdgeFitResult<Report> {
    let profile = load_profile(target_path)?;
    let metrics = analyze(&model, &profile);
    let policy = suppress_diagnostics(evaluate(&model, &profile, &metrics), suppressed_ids);
    Ok(build_report(&model, &profile, metrics, policy))
}
