//! EdgeFit Rust engine 的 Python ABI 边界。
//!
//! 绑定层只转换字符串、列表与异常；分析语义必须保留在 edgefit-core。

use edgefit_core::{
    render_adapter_generated_text, render_calibration_files_with_status, render_normalized_text,
    render_optimization_text, simulate_calibration_text, validate_target_text,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (model_json, target_yaml, target_source, format="json", suppressed_ids=None, adapter_generated=false))]
fn analyze(
    model_json: &str,
    target_yaml: &str,
    target_source: &str,
    format: &str,
    suppressed_ids: Option<Vec<String>>,
    adapter_generated: bool,
) -> PyResult<String> {
    let suppressed_ids = suppressed_ids.unwrap_or_default();
    let result = if adapter_generated {
        render_adapter_generated_text(
            model_json,
            target_yaml,
            target_source,
            &suppressed_ids,
            format,
        )
    } else {
        render_normalized_text(
            model_json,
            target_yaml,
            target_source,
            &suppressed_ids,
            format,
        )
    };
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
fn validate_target(target_yaml: &str, target_source: &str) -> PyResult<String> {
    validate_target_text(target_yaml, target_source).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (model_json, target_yaml, target_source, format="json", adapter_generated=false))]
fn optimize(
    model_json: &str,
    target_yaml: &str,
    target_source: &str,
    format: &str,
    adapter_generated: bool,
) -> PyResult<String> {
    render_optimization_text(
        model_json,
        target_yaml,
        target_source,
        format,
        adapter_generated,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (evidence, model, target, format="json"))]
fn verify_calibration(
    evidence: &str,
    model: &str,
    target: &str,
    format: &str,
) -> PyResult<(String, String)> {
    render_calibration_files_with_status(evidence, model, target, format)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (model_json, adapter_generated, source_model, target, scenario, out_dir))]
fn simulate_calibration(
    model_json: &str,
    adapter_generated: bool,
    source_model: &str,
    target: &str,
    scenario: &str,
    out_dir: &str,
) -> PyResult<(String, String)> {
    simulate_calibration_text(
        model_json,
        adapter_generated,
        source_model,
        target,
        scenario,
        out_dir,
    )
    .map(|result| (result.status, result.verification_json))
    .map_err(PyValueError::new_err)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze, module)?)?;
    module.add_function(wrap_pyfunction!(optimize, module)?)?;
    module.add_function(wrap_pyfunction!(validate_target, module)?)?;
    module.add_function(wrap_pyfunction!(verify_calibration, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_calibration, module)?)?;
    Ok(())
}
