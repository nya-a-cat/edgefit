//! EdgeFit 静态事实分析：生成可复核的模型指标，不在本层决定诊断严重级别。

mod memory_plan;

pub use memory_plan::{MemoryAllocationTrace, PeakMemoryContributor};

use edgefit_ir::{normalize_dtype, NormalizedModel, TensorInfo};
use edgefit_target::TargetProfile;
use memory_plan::{plan_activation_memory, plan_activation_memory_with_defaults};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct TensorRankViolation {
    pub tensor: String,
    pub rank: u64,
    pub max_rank: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpDtypeViolation {
    pub op_type: String,
    pub tensor: String,
    pub dtype: String,
    pub allowed: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpRankViolation {
    pub op_type: String,
    pub tensor: String,
    pub rank: u64,
    pub max_rank: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializerDtypeMetric {
    pub dtype: String,
    pub tensor_count: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphIoDtypeViolation {
    pub tensor: String,
    pub boundary: String,
    pub dtype: String,
}

#[derive(Default)]
struct QuantizationEvidence {
    qdq_covered_node_count: u64,
    qoperator_node_count: u64,
    eligible_node_count: u64,
    covered_node_indexes: BTreeSet<usize>,
    representation: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    pub model_file_bytes: u64,
    pub external_data_file_count: u64,
    pub opset_versions: BTreeMap<String, u64>,
    pub initializer_bytes: u64,
    pub unresolved_initializer_size_count: u64,
    pub estimated_peak_activation_bytes: u64,
    pub planned_activation_arena_bytes: u64,
    pub activation_tensor_alignment_bytes: u64,
    pub activation_planner_algorithm: String,
    pub activation_planning_overflowed: bool,
    pub peak_activation_event: String,
    pub peak_activation_node_index: Option<u64>,
    pub peak_activation_node_name: Option<String>,
    pub peak_activation_op_type: Option<String>,
    pub peak_activation_live_allocated_bytes: u64,
    pub peak_activation_workspace_bytes: u64,
    pub peak_activation_fragmentation_bytes: u64,
    pub inplace_reuse_count: u64,
    pub inplace_avoided_allocation_bytes: u64,
    pub peak_activation_contributors: Vec<PeakMemoryContributor>,
    pub activation_allocation_trace: Vec<MemoryAllocationTrace>,
    pub peak_activation_confidence: String,
    pub shape_inference_status: String,
    pub shape_inference_error: Option<String>,
    pub dynamic_tensor_count: u64,
    pub bounded_dynamic_tensor_count: u64,
    pub unresolved_tensor_size_count: u64,
    pub unsupported_ops: Vec<String>,
    pub unsupported_dtypes: Vec<String>,
    pub unknown_dtype_tensors: Vec<String>,
    pub tensor_rank_violations: Vec<TensorRankViolation>,
    pub op_dtype_violations: Vec<OpDtypeViolation>,
    pub op_rank_violations: Vec<OpRankViolation>,
    pub quantized_initializer_fraction: f64,
    pub initializer_dtype_distribution: Vec<InitializerDtypeMetric>,
    pub quantization_representation: String,
    pub qdq_covered_node_count: u64,
    pub qoperator_node_count: u64,
    pub quantization_eligible_node_count: u64,
    pub quantization_covered_node_count: u64,
    pub quantization_operator_coverage: f64,
    pub target_requires_int8: bool,
    pub int8_tensor_count: u64,
    pub uint8_tensor_count: u64,
    pub non_int8_graph_io: Vec<GraphIoDtypeViolation>,
    pub ops_without_int8_path: Vec<String>,
    pub node_count: u64,
    pub tensor_count: u64,
}

pub fn analyze(model: &NormalizedModel, profile: &TargetProfile) -> Metrics {
    let memory_plan = plan_activation_memory(model, profile);
    let initializer_bytes = model
        .initializers()
        .filter_map(|tensor| tensor.byte_size())
        .fold(0_u64, u64::saturating_add);
    let unresolved_initializer_size_count = model
        .initializers()
        .filter(|tensor| tensor.byte_size().is_none())
        .count() as u64;
    let quantized_initializer_bytes = model
        .initializers()
        .filter(|tensor| {
            tensor
                .dtype
                .as_deref()
                .map(normalize_dtype)
                .map(|dtype| dtype == "int8" || dtype == "uint8")
                .unwrap_or(false)
        })
        .filter_map(|tensor| tensor.byte_size())
        .fold(0_u64, u64::saturating_add);

    let unsupported_ops = model
        .nodes
        .iter()
        .filter(|node| profile.op_rule(&node.domain, &node.op_type).is_none())
        .map(|node| node.op_type.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let unsupported_dtypes = model
        .tensors
        .values()
        .filter_map(|tensor| tensor.dtype.as_deref())
        .map(normalize_dtype)
        .filter(|dtype| !profile.dtype_allowed.contains(dtype))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let unknown_dtype_tensors = model
        .tensors
        .values()
        .filter(|tensor| tensor.dtype.is_none())
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();

    let tensor_rank_violations = profile
        .shape_max_rank
        .map(|max_rank| global_rank_violations(model, max_rank))
        .unwrap_or_default();

    let op_dtype_violations = op_dtype_violations(model, profile);
    let op_rank_violations = op_rank_violations(model, profile);
    let initializer_dtype_distribution = initializer_dtype_distribution(model);
    let quantization_evidence = collect_quantization_evidence(model, profile);
    let int8_tensor_count = tensor_dtype_count(model, "int8");
    let uint8_tensor_count = tensor_dtype_count(model, "uint8");
    let target_requires_int8 = profile.require_int8;
    let non_int8_graph_io = non_int8_graph_io(model);
    let ops_without_int8_path = ops_without_int8_path(model, profile);

    let dynamic_tensor_count = model
        .non_initializers()
        .filter(|tensor| !tensor.has_static_shape())
        .count() as u64;

    Metrics {
        model_file_bytes: model.file_bytes,
        external_data_file_count: model.external_data_file_count,
        opset_versions: model.opset_versions.clone(),
        initializer_bytes,
        unresolved_initializer_size_count,
        estimated_peak_activation_bytes: memory_plan.logical_peak_bytes,
        planned_activation_arena_bytes: memory_plan.planned_arena_bytes,
        activation_tensor_alignment_bytes: memory_plan.alignment_bytes,
        activation_planner_algorithm: memory_plan.algorithm,
        activation_planning_overflowed: memory_plan.planning_overflowed,
        peak_activation_event: memory_plan.peak_event,
        peak_activation_node_index: memory_plan.peak_node_index,
        peak_activation_node_name: memory_plan.peak_node_name,
        peak_activation_op_type: memory_plan.peak_op_type,
        peak_activation_live_allocated_bytes: memory_plan.peak_live_allocated_bytes,
        peak_activation_workspace_bytes: memory_plan.peak_workspace_bytes,
        peak_activation_fragmentation_bytes: memory_plan.peak_fragmentation_bytes,
        inplace_reuse_count: memory_plan.inplace_reuse_count,
        inplace_avoided_allocation_bytes: memory_plan.inplace_avoided_allocation_bytes,
        peak_activation_contributors: memory_plan.peak_contributors,
        activation_allocation_trace: memory_plan.allocation_trace,
        peak_activation_confidence: memory_plan.confidence,
        shape_inference_status: model.shape_inference_status.clone(),
        shape_inference_error: model.shape_inference_error.clone(),
        dynamic_tensor_count,
        bounded_dynamic_tensor_count: memory_plan.bounded_dynamic_tensor_count,
        unresolved_tensor_size_count: memory_plan.unresolved_tensor_size_count,
        unsupported_ops,
        unsupported_dtypes,
        unknown_dtype_tensors,
        tensor_rank_violations,
        op_dtype_violations,
        op_rank_violations,
        quantized_initializer_fraction: if initializer_bytes == 0 {
            1.0
        } else {
            quantized_initializer_bytes as f64 / initializer_bytes as f64
        },
        initializer_dtype_distribution,
        quantization_representation: quantization_evidence.representation,
        qdq_covered_node_count: quantization_evidence.qdq_covered_node_count,
        qoperator_node_count: quantization_evidence.qoperator_node_count,
        quantization_eligible_node_count: quantization_evidence.eligible_node_count,
        quantization_covered_node_count: quantization_evidence.covered_node_indexes.len() as u64,
        quantization_operator_coverage: if quantization_evidence.eligible_node_count == 0 {
            0.0
        } else {
            quantization_evidence.covered_node_indexes.len() as f64
                / quantization_evidence.eligible_node_count as f64
        },
        target_requires_int8,
        int8_tensor_count,
        uint8_tensor_count,
        non_int8_graph_io,
        ops_without_int8_path,
        node_count: model.nodes.len() as u64,
        tensor_count: model.tensors.len() as u64,
    }
}

/// 汇总 initializer 的 dtype、张量数量和字节数，供策略层输出可审计的量化缺口。
fn initializer_dtype_distribution(model: &NormalizedModel) -> Vec<InitializerDtypeMetric> {
    let mut distribution = BTreeMap::<String, (u64, u64)>::new();
    for tensor in model.initializers() {
        let dtype = tensor
            .dtype
            .as_deref()
            .map(normalize_dtype)
            .unwrap_or_else(|| "unknown".to_string());
        let entry = distribution.entry(dtype).or_insert((0, 0));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.saturating_add(tensor.byte_size().unwrap_or(0));
    }

    distribution
        .into_iter()
        .map(|(dtype, (tensor_count, bytes))| InitializerDtypeMetric {
            dtype,
            tensor_count,
            bytes,
        })
        .collect()
}

/// 识别 QDQ/QOperator 的图结构证据；该结果不等同于具体 runtime 的算子支持证明。
fn collect_quantization_evidence(
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> QuantizationEvidence {
    let mut producers = BTreeMap::<String, usize>::new();
    let mut consumers = BTreeMap::<String, Vec<usize>>::new();
    for (index, node) in model.nodes.iter().enumerate() {
        for output in node.outputs.iter().filter(|name| !name.is_empty()) {
            producers.insert(output.clone(), index);
        }
        for input in node.inputs.iter().filter(|name| !name.is_empty()) {
            consumers.entry(input.clone()).or_default().push(index);
        }
    }

    // 表示类型来自原始图证据，不能因当前 target 不支持某个量化算子而被隐藏。
    let qoperator_node_indexes = model
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| is_qoperator(node))
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    let qdq_covered_node_indexes = model
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            !is_quantization_plumbing(node) && !is_qoperator(node)
        })
        .filter(|(_, node)| {
            let input_from_dequantize = node.inputs.iter().any(|input| {
                producers
                    .get(input)
                    .map(|producer| model.nodes[*producer].op_type == "DequantizeLinear")
                    .unwrap_or(false)
            });
            let output_to_quantize = node.outputs.iter().any(|output| {
                consumers
                    .get(output)
                    .map(|indexes| {
                        indexes
                            .iter()
                            .any(|consumer| model.nodes[*consumer].op_type == "QuantizeLinear")
                    })
                    .unwrap_or(false)
            });
            input_from_dequantize || output_to_quantize
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();

    let eligible_node_indexes = model
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            if is_quantization_plumbing(node) {
                return false;
            }
            profile
                .op_rule(&node.domain, &node.op_type)
                .map(|rule| {
                    rule.dtypes
                        .iter()
                        .any(|dtype| matches!(dtype.as_str(), "int8" | "uint8"))
                })
                .unwrap_or(false)
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();

    let raw_covered_node_indexes = qdq_covered_node_indexes
        .union(&qoperator_node_indexes)
        .copied()
        .collect::<BTreeSet<_>>();
    let covered_node_indexes = raw_covered_node_indexes
        .intersection(&eligible_node_indexes)
        .copied()
        .collect::<BTreeSet<_>>();
    let representation = match (
        qdq_covered_node_indexes.is_empty(),
        qoperator_node_indexes.is_empty(),
    ) {
        (false, false) => "mixed",
        (false, true) => "qdq",
        (true, false) => "qoperator",
        (true, true) => "none",
    }
    .to_string();

    QuantizationEvidence {
        qdq_covered_node_count: qdq_covered_node_indexes.len() as u64,
        qoperator_node_count: qoperator_node_indexes.len() as u64,
        eligible_node_count: eligible_node_indexes.len() as u64,
        covered_node_indexes,
        representation,
    }
}

/// QOperator 只匹配 ONNX 标准算子和已知 ORT 扩展，不把任意自定义 Q 前缀当作量化证据。
fn is_qoperator(node: &edgefit_ir::NodeInfo) -> bool {
    if matches!(node.domain.as_str(), "" | "ai.onnx") {
        return matches!(
            node.op_type.as_str(),
            "QLinearConv" | "QLinearMatMul" | "ConvInteger" | "MatMulInteger"
        );
    }
    if node.domain == "com.microsoft" {
        return matches!(
            node.op_type.as_str(),
            "QAttention"
                | "QGemm"
                | "QLinearAdd"
                | "QLinearAveragePool"
                | "QLinearConcat"
                | "QLinearConv"
                | "QLinearGlobalAveragePool"
                | "QLinearLeakyRelu"
                | "QLinearMul"
                | "QLinearReduceMean"
                | "QLinearSigmoid"
                | "QLinearSoftmax"
                | "QLinearWhere"
                | "DynamicQuantizeLSTM"
                | "DynamicQuantizeMatMul"
                | "MatMulIntegerToFloat"
                | "MatMulNBits"
                | "GatherBlockQuantized"
        );
    }
    false
}

fn is_quantization_plumbing(node: &edgefit_ir::NodeInfo) -> bool {
    matches!(node.domain.as_str(), "" | "ai.onnx")
        && matches!(
            node.op_type.as_str(),
            "QuantizeLinear" | "DequantizeLinear" | "DynamicQuantizeLinear"
        )
}

fn tensor_dtype_count(model: &NormalizedModel, expected: &str) -> u64 {
    model
        .tensors
        .values()
        .filter_map(|tensor| tensor.dtype.as_deref())
        .map(normalize_dtype)
        .filter(|dtype| dtype == expected)
        .count() as u64
}

/// 记录严格 int8 目标需要处理的浮点或未知 graph I/O 边界。
fn non_int8_graph_io(model: &NormalizedModel) -> Vec<GraphIoDtypeViolation> {
    let mut violations = Vec::new();
    for (boundary, names) in [("input", &model.inputs), ("output", &model.outputs)] {
        for name in names {
            let dtype = model
                .tensors
                .get(name)
                .and_then(|tensor| tensor.dtype.as_deref())
                .map(normalize_dtype)
                .unwrap_or_else(|| "unknown".to_string());
            if !matches!(dtype.as_str(), "int8" | "uint8") {
                violations.push(GraphIoDtypeViolation {
                    tensor: name.clone(),
                    boundary: boundary.to_string(),
                    dtype,
                });
            }
        }
    }
    violations
}

/// 按 target op rule 识别没有 int8/uint8 kernel 路径的已用算子。
fn ops_without_int8_path(model: &NormalizedModel, profile: &TargetProfile) -> Vec<String> {
    model
        .nodes
        .iter()
        .filter_map(|node| {
            let rule = profile.op_rule(&node.domain, &node.op_type)?;
            if !rule
                .dtypes
                .iter()
                .any(|dtype| matches!(dtype.as_str(), "int8" | "uint8"))
            {
                Some(format!("{}::{}", node.domain, node.op_type))
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn global_rank_violations(model: &NormalizedModel, max_rank: u64) -> Vec<TensorRankViolation> {
    model
        .tensors
        .values()
        .filter_map(|tensor| {
            tensor_rank(tensor).and_then(|rank| {
                if rank > max_rank {
                    Some(TensorRankViolation {
                        tensor: tensor.name.clone(),
                        rank,
                        max_rank,
                    })
                } else {
                    None
                }
            })
        })
        .collect()
}

fn op_dtype_violations(model: &NormalizedModel, profile: &TargetProfile) -> Vec<OpDtypeViolation> {
    let mut violations = BTreeMap::new();
    for node in &model.nodes {
        let Some(rule) = profile.op_rule(&node.domain, &node.op_type) else {
            continue;
        };
        if rule.dtypes.is_empty() {
            continue;
        }
        for tensor_name in node.inputs.iter().chain(node.outputs.iter()) {
            let Some(tensor) = model.tensors.get(tensor_name) else {
                continue;
            };
            let Some(dtype) = tensor.dtype.as_deref().map(normalize_dtype) else {
                continue;
            };
            if !rule.dtypes.contains(&dtype) {
                violations.insert(
                    (node.op_type.clone(), tensor.name.clone(), dtype.clone()),
                    OpDtypeViolation {
                        op_type: node.op_type.clone(),
                        tensor: tensor.name.clone(),
                        dtype,
                        allowed: rule.dtypes.iter().cloned().collect(),
                    },
                );
            }
        }
    }
    violations.into_values().collect()
}

fn op_rank_violations(model: &NormalizedModel, profile: &TargetProfile) -> Vec<OpRankViolation> {
    let mut violations = BTreeMap::new();
    for node in &model.nodes {
        let Some(rule) = profile.op_rule(&node.domain, &node.op_type) else {
            continue;
        };
        let Some(max_rank) = rule.max_rank else {
            continue;
        };
        for tensor_name in node.inputs.iter().chain(node.outputs.iter()) {
            let Some(tensor) = model.tensors.get(tensor_name) else {
                continue;
            };
            let Some(rank) = tensor_rank(tensor) else {
                continue;
            };
            if rank > max_rank {
                violations.insert(
                    (node.op_type.clone(), tensor.name.clone(), rank),
                    OpRankViolation {
                        op_type: node.op_type.clone(),
                        tensor: tensor.name.clone(),
                        rank,
                        max_rank,
                    },
                );
            }
        }
    }
    violations.into_values().collect()
}

fn tensor_rank(tensor: &TensorInfo) -> Option<u64> {
    tensor.shape.as_ref().map(|shape| shape.len() as u64)
}

pub fn estimate_peak_activation(
    model: &NormalizedModel,
    symbol_bounds: &BTreeMap<String, u64>,
) -> (u64, String, u64, u64) {
    let plan = plan_activation_memory_with_defaults(model, symbol_bounds);
    (
        plan.logical_peak_bytes,
        plan.confidence,
        plan.bounded_dynamic_tensor_count,
        plan.unresolved_tensor_size_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefit_ir::{Dim, NodeInfo, TensorInfo};
    use edgefit_target::{OpRule, ProfileMetadata, TargetProfile};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    fn test_profile(allowed_ops: BTreeMap<(String, String), OpRule>) -> TargetProfile {
        TargetProfile {
            source: PathBuf::new(),
            fingerprint: "test".to_string(),
            metadata: ProfileMetadata::unknown(),
            target_id: "t".to_string(),
            target_name: None,
            target_class: None,
            flash_bytes: None,
            ram_bytes: None,
            model_file_budget_bytes: None,
            peak_activation_budget_bytes: None,
            weights_residency: None,
            tensor_alignment_bytes: 1,
            shape_max_rank: Some(4),
            allow_unknown_dims: false,
            symbol_bounds: BTreeMap::new(),
            runtime_name: None,
            static_shapes_required: true,
            dynamic_allocation_allowed: None,
            external_memory_allowed: None,
            dtype_allowed: BTreeSet::from(["int8".to_string()]),
            dtype_preferred: None,
            fp32_allowed: None,
            max_opset_versions: BTreeMap::new(),
            allowed_ops,
            quantization_required: false,
            require_int8: false,
            min_quantized_weight_fraction: None,
            min_quantized_operator_coverage: None,
        }
    }

    #[test]
    fn estimates_peak_activation() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "x".to_string(),
            TensorInfo {
                name: "x".to_string(),
                dtype: Some("int8".to_string()),
                shape: Some(vec![Dim::Known(1), Dim::Known(10)]),
                bytes: None,
                initializer: false,
            },
        );
        tensors.insert(
            "y".to_string(),
            TensorInfo {
                name: "y".to_string(),
                dtype: Some("int8".to_string()),
                shape: Some(vec![Dim::Known(1), Dim::Known(20)]),
                bytes: None,
                initializer: false,
            },
        );
        let model = NormalizedModel {
            path: "m".to_string(),
            sha256: "s".to_string(),
            file_bytes: 1,
            external_data_file_count: 0,
            adapter_generated: false,
            opset_versions: BTreeMap::new(),
            shape_inference_status: "not_recorded".to_string(),
            shape_inference_error: None,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            tensors,
            nodes: vec![NodeInfo {
                name: None,
                domain: "ai.onnx".to_string(),
                op_type: "Relu".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
            }],
        };
        let (peak, confidence, _, _) = estimate_peak_activation(&model, &BTreeMap::new());
        assert_eq!(peak, 30);
        assert_eq!(confidence, "high");
    }

    #[test]
    fn reports_unsupported_op() {
        let model = NormalizedModel {
            path: "m".to_string(),
            sha256: "s".to_string(),
            file_bytes: 1,
            external_data_file_count: 0,
            adapter_generated: false,
            opset_versions: BTreeMap::new(),
            shape_inference_status: "not_recorded".to_string(),
            shape_inference_error: None,
            inputs: vec![],
            outputs: vec![],
            tensors: BTreeMap::new(),
            nodes: vec![NodeInfo {
                name: None,
                domain: "ai.onnx".to_string(),
                op_type: "Resize".to_string(),
                inputs: vec![],
                outputs: vec![],
            }],
        };
        let mut allowed_ops = BTreeMap::new();
        allowed_ops.insert(
            ("ai.onnx".to_string(), "Relu".to_string()),
            OpRule {
                dtypes: BTreeSet::from(["int8".to_string()]),
                max_rank: None,
                workspace_bytes: 0,
                first_output_inplace_input_index: None,
            },
        );
        let metrics = analyze(&model, &test_profile(allowed_ops));
        assert_eq!(metrics.unsupported_ops, vec!["Resize"]);
    }

    #[test]
    fn reports_rank_and_op_dtype_violations() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "x".to_string(),
            TensorInfo {
                name: "x".to_string(),
                dtype: Some("float32".to_string()),
                shape: Some(vec![
                    Dim::Known(1),
                    Dim::Known(2),
                    Dim::Known(3),
                    Dim::Known(4),
                    Dim::Known(5),
                ]),
                bytes: None,
                initializer: false,
            },
        );
        let model = NormalizedModel {
            path: "m".to_string(),
            sha256: "s".to_string(),
            file_bytes: 1,
            external_data_file_count: 0,
            adapter_generated: false,
            opset_versions: BTreeMap::new(),
            shape_inference_status: "not_recorded".to_string(),
            shape_inference_error: None,
            inputs: vec!["x".to_string()],
            outputs: vec!["x".to_string()],
            tensors,
            nodes: vec![NodeInfo {
                name: None,
                domain: "ai.onnx".to_string(),
                op_type: "Conv".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["x".to_string()],
            }],
        };
        let mut allowed_ops = BTreeMap::new();
        allowed_ops.insert(
            ("ai.onnx".to_string(), "Conv".to_string()),
            OpRule {
                dtypes: BTreeSet::from(["int8".to_string()]),
                max_rank: Some(4),
                workspace_bytes: 0,
                first_output_inplace_input_index: None,
            },
        );
        let metrics = analyze(&model, &test_profile(allowed_ops));
        assert_eq!(metrics.tensor_rank_violations.len(), 1);
        assert_eq!(metrics.op_dtype_violations.len(), 1);
        assert_eq!(metrics.op_rank_violations.len(), 1);
    }

    #[test]
    fn plans_alignment_workspace_and_safe_inplace_reuse() {
        let tensors = ["x", "y", "z"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    TensorInfo {
                        name: name.to_string(),
                        dtype: Some("int8".to_string()),
                        shape: Some(vec![Dim::Known(10)]),
                        bytes: None,
                        initializer: false,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let model = NormalizedModel {
            path: "m".to_string(),
            sha256: "s".to_string(),
            file_bytes: 1,
            external_data_file_count: 0,
            adapter_generated: false,
            opset_versions: BTreeMap::new(),
            shape_inference_status: "not_recorded".to_string(),
            shape_inference_error: None,
            inputs: vec!["x".to_string()],
            outputs: vec!["z".to_string()],
            tensors,
            nodes: vec![
                NodeInfo {
                    name: Some("first".to_string()),
                    domain: "ai.onnx".to_string(),
                    op_type: "Relu".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                },
                NodeInfo {
                    name: Some("second".to_string()),
                    domain: "ai.onnx".to_string(),
                    op_type: "Relu".to_string(),
                    inputs: vec!["y".to_string(), "y".to_string()],
                    outputs: vec!["z".to_string()],
                },
            ],
        };
        let mut profile = test_profile(BTreeMap::from([(
            ("ai.onnx".to_string(), "Relu".to_string()),
            OpRule {
                dtypes: BTreeSet::from(["int8".to_string()]),
                max_rank: None,
                workspace_bytes: 8,
                first_output_inplace_input_index: Some(0),
            },
        )]));
        profile.tensor_alignment_bytes = 16;

        let metrics = analyze(&model, &profile);

        assert_eq!(metrics.estimated_peak_activation_bytes, 20);
        assert_eq!(metrics.planned_activation_arena_bytes, 48);
        assert_eq!(metrics.peak_activation_workspace_bytes, 16);
        assert_eq!(metrics.inplace_reuse_count, 1);
        assert_eq!(metrics.inplace_avoided_allocation_bytes, 16);
        assert!(metrics
            .activation_allocation_trace
            .iter()
            .any(|item| item.name == "z" && item.alias_of.as_deref() == Some("y")));
    }

    #[test]
    fn reports_planner_overflow_without_relabeling_tensor_size_as_unknown() {
        let model = NormalizedModel {
            path: "m".to_string(),
            sha256: "s".to_string(),
            file_bytes: 1,
            external_data_file_count: 0,
            adapter_generated: false,
            opset_versions: BTreeMap::new(),
            shape_inference_status: "not_recorded".to_string(),
            shape_inference_error: None,
            inputs: vec!["x".to_string()],
            outputs: vec!["x".to_string()],
            tensors: BTreeMap::from([(
                "x".to_string(),
                TensorInfo {
                    name: "x".to_string(),
                    dtype: Some("int8".to_string()),
                    shape: Some(vec![Dim::Known(1)]),
                    bytes: Some(u64::MAX - 1),
                    initializer: false,
                },
            )]),
            nodes: vec![],
        };
        let mut profile = test_profile(BTreeMap::new());
        profile.tensor_alignment_bytes = 4;

        let metrics = analyze(&model, &profile);

        assert!(metrics.activation_planning_overflowed);
        assert_eq!(metrics.unresolved_tensor_size_count, 0);
        assert_eq!(metrics.planned_activation_arena_bytes, u64::MAX);
    }
}
