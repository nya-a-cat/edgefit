//! EdgeFit Rust engine 的 Python ABI 边界。
//!
//! 绑定层只转换字符串、列表与异常；分析语义必须保留在 edgefit-core。

use edgefit_core::{
    render_adapter_generated_text, render_normalized_text, render_optimization_text,
    validate_target_text,
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze, module)?)?;
    module.add_function(wrap_pyfunction!(optimize, module)?)?;
    module.add_function(wrap_pyfunction!(validate_target, module)?)?;
    Ok(())
}
