//! EdgeFit 目标配置的领域模型、严格解析与业务约束校验。
//!
//! 本模块保持 profile 的向后兼容默认值，并拒绝无法安全解释的字段或取值。

use edgefit_ir::{AttributeValue, EdgeFitResult, FloatAttribute};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpRule {
    pub dtypes: BTreeSet<String>,
    /// 按零基输入端口声明的 dtype 白名单；未声明端口时沿用聚合 dtypes。
    pub input_dtypes: BTreeMap<usize, BTreeSet<String>>,
    /// 按零基输出端口声明的 dtype 白名单；未声明端口时沿用聚合 dtypes。
    pub output_dtypes: BTreeMap<usize, BTreeSet<String>>,
    /// 属性名到允许标量值的白名单；未声明属性时不增加属性约束。
    pub attributes: BTreeMap<String, BTreeSet<AttributeValue>>,
    pub max_rank: Option<u64>,
    /// 单个节点执行期间需要独占的临时工作区字节数。
    pub workspace_bytes: u64,
    /// 可被首个输出安全复用的输入索引；未声明时禁止原地复用。
    pub first_output_inplace_input_index: Option<usize>,
    /// CPU 基线代价；缺失时不得生成总延迟。
    pub cpu_cost: Option<KernelCost>,
    /// 可选 NPU kernel 代价；存在时该节点可成为 NPU 分区候选。
    pub npu_cost: Option<KernelCost>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelCost {
    pub id: String,
    pub kind: String,
    pub fixed_ns: u64,
    pub throughput_per_second: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceleratorProfile {
    pub id: String,
    pub confidence: String,
    pub scratchpad_bytes: u64,
    pub tensor_alignment_bytes: u64,
    pub dma_burst_bytes: u64,
    pub dma_setup_ns: u64,
    pub dma_read_bytes_per_second: u64,
    pub dma_write_bytes_per_second: u64,
    pub spill_allowed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplacementRecipe {
    pub id: String,
    pub trusted: bool,
    pub source: String,
    pub version: String,
    pub replacement_ops: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileMetadata {
    pub source: String,
    pub confidence: String,
    pub last_verified: String,
}

impl ProfileMetadata {
    pub fn unknown() -> Self {
        Self {
            source: "unknown".to_string(),
            confidence: "unknown".to_string(),
            last_verified: "unknown".to_string(),
        }
    }

    fn is_complete(&self) -> bool {
        !self.source.trim().is_empty()
            && !self.confidence.trim().is_empty()
            && !self.last_verified.trim().is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetProfile {
    pub source: PathBuf,
    pub fingerprint: String,
    pub metadata: ProfileMetadata,
    pub target_id: String,
    pub target_name: Option<String>,
    pub target_class: Option<String>,
    pub flash_bytes: Option<u64>,
    pub ram_bytes: Option<u64>,
    pub model_file_budget_bytes: Option<u64>,
    pub peak_activation_budget_bytes: Option<u64>,
    pub weights_residency: Option<String>,
    /// 张量分配的全局字节对齐；旧 profile 默认按 1 字节对齐。
    pub tensor_alignment_bytes: u64,
    pub shape_max_rank: Option<u64>,
    pub allow_unknown_dims: bool,
    pub symbol_bounds: BTreeMap<String, u64>,
    pub runtime_name: Option<String>,
    pub static_shapes_required: bool,
    pub dynamic_allocation_allowed: Option<bool>,
    pub external_memory_allowed: Option<bool>,
    pub dtype_allowed: BTreeSet<String>,
    pub dtype_preferred: Option<String>,
    pub fp32_allowed: Option<bool>,
    pub max_opset_versions: BTreeMap<String, u64>,
    pub allowed_ops: BTreeMap<(String, String), OpRule>,
    pub accelerator: Option<AcceleratorProfile>,
    pub replacement_recipes: BTreeMap<(String, String), ReplacementRecipe>,
    pub quantization_required: bool,
    pub require_int8: bool,
    pub min_quantized_weight_fraction: Option<f64>,
    pub min_quantized_operator_coverage: Option<f64>,
}

impl TargetProfile {
    pub fn op_rule(&self, domain: &str, op_type: &str) -> Option<&OpRule> {
        self.allowed_ops
            .get(&(domain.to_string(), op_type.to_string()))
    }

    pub fn validate(&self) -> EdgeFitResult<()> {
        let mut errors = Vec::new();
        if !self.metadata.is_complete() {
            errors.push(
                "metadata.source, metadata.confidence, and metadata.last_verified are required"
                    .to_string(),
            );
        }
        if self.target_id.trim().is_empty() {
            errors.push("target.id is required".to_string());
        }
        if self.fingerprint.trim().is_empty() {
            errors.push("target profile fingerprint is required".to_string());
        }
        if self
            .target_name
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            errors.push("target.name is required".to_string());
        }
        if self
            .target_class
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            errors.push("target.class is required".to_string());
        }
        if let (Some(budget), Some(capacity)) = (self.model_file_budget_bytes, self.flash_bytes) {
            if budget > capacity {
                errors.push("memory.model_file_budget_bytes cannot exceed memory.flash_bytes".to_string());
            }
        }
        if let (Some(budget), Some(capacity)) = (self.peak_activation_budget_bytes, self.ram_bytes) {
            if budget > capacity {
                errors.push(
                    "memory.peak_activation_budget_bytes cannot exceed memory.ram_bytes"
                        .to_string(),
                );
            }
        }
        if let Some(residency) = self.weights_residency.as_deref() {
            if !matches!(residency, "flash" | "file" | "ram") {
                errors.push("memory.weights_residency must be flash, file, or ram".to_string());
            }
        } else {
            errors.push("memory.weights_residency is required".to_string());
        }
        if self.model_file_budget_bytes.is_none() || self.peak_activation_budget_bytes.is_none() {
            errors.push(
                "memory.model_file_budget_bytes and memory.peak_activation_budget_bytes are required"
                    .to_string(),
            );
        }
        if self.flash_bytes.is_none() || self.ram_bytes.is_none() {
            errors.push("memory.flash_bytes and memory.ram_bytes are required".to_string());
        }
        if self.weights_residency.as_deref() == Some("ram") && self.ram_bytes.is_none() {
            errors.push("memory.weights_residency=ram requires memory.ram_bytes".to_string());
        }
        if self.tensor_alignment_bytes == 0 || !self.tensor_alignment_bytes.is_power_of_two() {
            errors.push(
                "memory.tensor_alignment_bytes must be greater than zero and a power of two"
                    .to_string(),
            );
        }
        if self
            .runtime_name
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            errors.push("runtime.name is required".to_string());
        }
        if self.static_shapes_required && self.allow_unknown_dims {
            errors.push(
                "runtime.static_shapes_required=true requires shape.allow_unknown_dims=false"
                    .to_string(),
            );
        }
        if self.dynamic_allocation_allowed.is_none() || self.external_memory_allowed.is_none() {
            errors.push(
                "runtime.dynamic_allocation_allowed and runtime.external_memory_allowed are required"
                    .to_string(),
            );
        }
        if self.dtype_allowed.is_empty() {
            errors.push("dtype.allowed must contain at least one dtype".to_string());
        }
        if self.dtype_preferred.is_none() || self.fp32_allowed.is_none() {
            errors.push("dtype.preferred and dtype.fp32_allowed are required".to_string());
        }
        if self.shape_max_rank.is_none() {
            errors.push("shape.max_rank is required".to_string());
        }
        if let Some(preferred) = &self.dtype_preferred {
            if !self.dtype_allowed.contains(preferred) {
                errors.push("dtype.preferred must also appear in dtype.allowed".to_string());
            }
        }
        if let Some(fp32_allowed) = self.fp32_allowed {
            if fp32_allowed != self.dtype_allowed.contains("float32") {
                errors.push(
                    "dtype.fp32_allowed must match whether float32 appears in dtype.allowed"
                        .to_string(),
                );
            }
        }
        for (symbol, maximum) in &self.symbol_bounds {
            if *maximum == 0 {
                errors.push(format!("shape.symbols.{symbol}.max must be greater than zero"));
            }
        }
        if self.allowed_ops.is_empty() {
            errors.push("ops.allow must contain at least one operator".to_string());
        }
        for ((domain, op), rule) in &self.allowed_ops {
            if rule.dtypes.is_empty() {
                errors.push(format!(
                    "ops.allow.{domain}.{op}.dtypes must contain at least one dtype"
                ));
            }
            if let Some(ram_bytes) = self.ram_bytes {
                if rule.workspace_bytes > ram_bytes {
                    errors.push(format!(
                        "ops.allow.{domain}.{op}.workspace_bytes cannot exceed memory.ram_bytes"
                    ));
                }
            }
            for (scope, contracts) in [
                ("input_dtypes", &rule.input_dtypes),
                ("output_dtypes", &rule.output_dtypes),
            ] {
                for (port, allowed) in contracts {
                    if allowed.is_empty() {
                        errors.push(format!(
                            "ops.allow.{domain}.{op}.{scope}.{port} must contain at least one dtype"
                        ));
                    }
                }
            }
            for (attribute, allowed) in &rule.attributes {
                if allowed.is_empty() {
                    errors.push(format!(
                        "ops.allow.{domain}.{op}.attributes.{attribute} must contain at least one scalar value"
                    ));
                }
            }
            for (scope, cost) in [("cpu_cost", &rule.cpu_cost), ("npu_cost", &rule.npu_cost)] {
                if let Some(cost) = cost {
                    validate_kernel_cost(domain, op, scope, cost, &mut errors);
                }
            }
        }
        if let Some(accelerator) = &self.accelerator {
            if accelerator.id.trim().is_empty() || accelerator.confidence.trim().is_empty() {
                errors.push("accelerator.id and accelerator.confidence are required".to_string());
            }
            if accelerator.scratchpad_bytes == 0 {
                errors.push("accelerator.scratchpad_bytes must be greater than zero".to_string());
            }
            for (name, value) in [
                ("tensor_alignment_bytes", accelerator.tensor_alignment_bytes),
                ("dma_burst_bytes", accelerator.dma_burst_bytes),
            ] {
                if value == 0 || !value.is_power_of_two() {
                    errors.push(format!("accelerator.{name} must be a non-zero power of two"));
                }
            }
            if accelerator.dma_read_bytes_per_second == 0
                || accelerator.dma_write_bytes_per_second == 0
            {
                errors.push("accelerator DMA bandwidths must be greater than zero".to_string());
            }
        } else if self
            .allowed_ops
            .values()
            .any(|rule| rule.npu_cost.is_some())
            || !self.replacement_recipes.is_empty()
        {
            errors.push("NPU costs and recipes require an accelerator section".to_string());
        }
        for ((domain, op), recipe) in &self.replacement_recipes {
            if !recipe.trusted
                || recipe.id.trim().is_empty()
                || recipe.source.trim().is_empty()
                || recipe.version.trim().is_empty()
                || recipe.replacement_ops.is_empty()
            {
                errors.push(format!(
                    "recipes.{domain}.{op} requires trusted=true, id, source, version, and replacement_ops"
                ));
            }
            for replacement in &recipe.replacement_ops {
                if self
                    .op_rule(domain, replacement)
                    .and_then(|rule| rule.npu_cost.as_ref())
                    .is_none()
                {
                    errors.push(format!(
                        "recipes.{domain}.{op} replacement {replacement} requires an NPU cost"
                    ));
                }
            }
        }
        // int8 是量化策略的收窄条件，不能脱离 quantization.required 单独启用。
        if self.require_int8 && !self.quantization_required {
            errors.push("quantization.require_int8 requires quantization.required=true".to_string());
        }
        if !self.quantization_required
            && (self.min_quantized_weight_fraction.unwrap_or(0.0) > 0.0
                || self.min_quantized_operator_coverage.unwrap_or(0.0) > 0.0)
        {
            errors.push(
                "positive quantization minimums require quantization.required=true".to_string(),
            );
        }
        if self.require_int8
            && !self.dtype_allowed.contains("int8")
            && !self.dtype_allowed.contains("uint8")
        {
            errors.push(
                "quantization.require_int8 requires int8 or uint8 in dtype.allowed".to_string(),
            );
        }
        for (field, value) in [
            (
                "quantization.min_quantized_weight_fraction",
                self.min_quantized_weight_fraction,
            ),
            (
                "quantization.min_operator_coverage",
                self.min_quantized_operator_coverage,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    errors.push(format!("{field} must be between 0 and 1"));
                }
            }
        }
        if self.min_quantized_weight_fraction.is_none()
            || self.min_quantized_operator_coverage.is_none()
        {
            errors.push(
                "quantization.min_quantized_weight_fraction and quantization.min_operator_coverage are required"
                    .to_string(),
            );
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "invalid target profile:\n- {}",
                errors.join("\n- ")
            ))
        }
    }
}

fn validate_kernel_cost(
    domain: &str,
    op: &str,
    scope: &str,
    cost: &KernelCost,
    errors: &mut Vec<String>,
) {
    if cost.id.trim().is_empty() || !matches!(cost.kind.as_str(), "fixed" | "mac" | "element" | "bytes") {
        errors.push(format!("ops.allow.{domain}.{op}.{scope} requires id and a supported kind"));
    }
    if cost.kind != "fixed" && cost.throughput_per_second == 0 {
        errors.push(format!(
            "ops.allow.{domain}.{op}.{scope}.throughput_per_second must be greater than zero"
        ));
    }
}

#[derive(Default)]
struct ParseState {
    section: String,
    ops_allow: bool,
    ops_domain: String,
    current_op: Option<String>,
    op_rule_section: Option<String>,
    shape_symbols: bool,
    recipe_domain: String,
    current_recipe_op: Option<String>,
}

pub fn load_profile(path: impl AsRef<Path>) -> EdgeFitResult<TargetProfile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read profile: {err}"))?;
    let profile = parse_profile(&text, path.to_path_buf())?;
    profile.validate()?;
    Ok(profile)
}

pub fn parse_profile(text: &str, source: PathBuf) -> EdgeFitResult<TargetProfile> {
    let mut state = ParseState::default();
    let mut profile_version = String::new();
    let mut metadata = ProfileMetadata {
        source: String::new(),
        confidence: String::new(),
        last_verified: String::new(),
    };
    let mut target_id = String::new();
    let mut target_name = None;
    let mut target_class = None;
    let mut flash_bytes = None;
    let mut ram_bytes = None;
    let mut model_file_budget_bytes = None;
    let mut peak_activation_budget_bytes = None;
    let mut weights_residency = None;
    let mut tensor_alignment_bytes = 1;
    let mut shape_max_rank = None;
    let mut allow_unknown_dims = true;
    let mut symbol_bounds = BTreeMap::new();
    let mut static_shapes_required = false;
    let mut runtime_name = None;
    let mut dynamic_allocation_allowed = None;
    let mut external_memory_allowed = None;
    let mut dtype_allowed = BTreeSet::new();
    let mut dtype_preferred = None;
    let mut fp32_allowed = None;
    let mut max_opset_versions = BTreeMap::new();
    let mut allowed_ops = BTreeMap::new();
    let mut accelerator_section_present = false;
    let mut accelerator_id = None;
    let mut accelerator_confidence = None;
    let mut accelerator_scratchpad_bytes = None;
    let mut accelerator_tensor_alignment_bytes = None;
    let mut accelerator_dma_burst_bytes = None;
    let mut accelerator_dma_setup_ns = None;
    let mut accelerator_dma_read_bytes_per_second = None;
    let mut accelerator_dma_write_bytes_per_second = None;
    let mut accelerator_spill_allowed = None;
    let mut replacement_recipes = BTreeMap::new();
    let mut quantization_required = false;
    let mut require_int8 = false;
    let mut min_quantized_weight_fraction = None;
    let mut min_quantized_operator_coverage = None;

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count();
        let trimmed = line.trim();
        let Some((key, raw_value)) = trimmed.split_once(':') else {
            return Err(format!("unsupported profile line: {raw_line}"));
        };
        let key = key.trim().trim_start_matches('\u{feff}');
        let value = raw_value.trim();

        if indent == 0 {
            if !matches!(
                key,
                "profile_version"
                    | "metadata"
                    | "target"
                    | "memory"
                    | "runtime"
                    | "dtype"
                    | "opsets"
                    | "ops"
                    | "shape"
                    | "quantization"
                    | "accelerator"
                    | "recipes"
            ) {
                return Err(format!("unsupported profile section {key}"));
            }
            state.section = key.to_string();
            if key == "accelerator" {
                accelerator_section_present = true;
            }
            state.ops_allow = false;
            state.ops_domain.clear();
            state.current_op = None;
            state.op_rule_section = None;
            state.shape_symbols = false;
            state.recipe_domain.clear();
            state.current_recipe_op = None;
            if key == "profile_version" {
                profile_version = clean_scalar(value);
            }
            continue;
        }

        match state.section.as_str() {
            "metadata" if indent == 2 && key == "source" => metadata.source = clean_scalar(value),
            "metadata" if indent == 2 && key == "confidence" => {
                metadata.confidence = clean_scalar(value);
            }
            "metadata" if indent == 2 && key == "last_verified" => {
                metadata.last_verified = clean_scalar(value);
            }
            "target" if indent == 2 && key == "id" => target_id = clean_scalar(value),
            "target" if indent == 2 && key == "name" => {
                target_name = Some(clean_scalar(value));
            }
            "target" if indent == 2 && key == "class" => {
                target_class = Some(clean_scalar(value));
            }
            "memory" if indent == 2 && key == "flash_bytes" => {
                flash_bytes = parse_u64(value, key)?;
            }
            "memory" if indent == 2 && key == "ram_bytes" => {
                ram_bytes = parse_u64(value, key)?;
            }
            "memory" if indent == 2 && key == "model_file_budget_bytes" => {
                model_file_budget_bytes = parse_u64(value, key)?;
            }
            "memory" if indent == 2 && key == "peak_activation_budget_bytes" => {
                peak_activation_budget_bytes = parse_u64(value, key)?;
            }
            "memory" if indent == 2 && key == "weights_residency" => {
                weights_residency = Some(clean_scalar(value));
            }
            "memory" if indent == 2 && key == "tensor_alignment_bytes" => {
                tensor_alignment_bytes = parse_u64(value, key)?.ok_or_else(|| {
                    "memory.tensor_alignment_bytes requires an integer".to_string()
                })?;
            }
            "runtime" if indent == 2 && key == "name" => {
                runtime_name = Some(clean_scalar(value));
            }
            "runtime" if indent == 2 && key == "static_shapes_required" => {
                static_shapes_required = parse_bool(value, key)?;
            }
            "runtime" if indent == 2 && key == "dynamic_allocation_allowed" => {
                dynamic_allocation_allowed = Some(parse_bool(value, key)?);
            }
            "runtime" if indent == 2 && key == "external_memory_allowed" => {
                external_memory_allowed = Some(parse_bool(value, key)?);
            }
            "shape" if indent == 2 && key == "max_rank" => {
                shape_max_rank = parse_u64(value, key)?;
            }
            "shape" if indent == 2 && key == "allow_unknown_dims" => {
                allow_unknown_dims = parse_bool(value, key)?;
            }
            "shape" if indent == 2 && key == "symbols" => {
                state.shape_symbols = true;
            }
            "shape" if indent == 4 && state.shape_symbols => {
                if let Some(max) = parse_symbol_max(value)? {
                    symbol_bounds.insert(key.to_string(), max);
                }
            }
            "dtype" if indent == 2 && key == "allowed" => {
                dtype_allowed = parse_list(value)
                    .into_iter()
                    .map(|item| edgefit_ir::normalize_dtype(&item))
                    .collect();
            }
            "dtype" if indent == 2 && key == "preferred" => {
                dtype_preferred = Some(edgefit_ir::normalize_dtype(&clean_scalar(value)));
            }
            "dtype" if indent == 2 && key == "fp32_allowed" => {
                fp32_allowed = Some(parse_bool(value, key)?);
            }
            "opsets" if indent == 2 => {
                let version = parse_u64(value, key)?
                    .ok_or_else(|| format!("opsets.{key} requires an integer version"))?;
                if version == 0 {
                    return Err(format!("opsets.{key} must be greater than zero"));
                }
                let domain = if key.is_empty() || key == "ai.onnx" {
                    "ai.onnx".to_string()
                } else {
                    key.to_string()
                };
                if max_opset_versions.insert(domain.clone(), version).is_some() {
                    return Err(format!("duplicate opset limit for domain {domain}"));
                }
            }
            "quantization" if indent == 2 && key == "required" => {
                quantization_required = parse_bool(value, key)?;
            }
            "quantization" if indent == 2 && key == "require_int8" => {
                require_int8 = parse_bool(value, key)?;
            }
            "quantization" if indent == 2 && key == "min_quantized_weight_fraction" => {
                min_quantized_weight_fraction = Some(parse_f64(value, key)?);
            }
            "quantization" if indent == 2 && key == "min_operator_coverage" => {
                min_quantized_operator_coverage = Some(parse_f64(value, key)?);
            }
            "accelerator" if indent == 2 && key == "id" => {
                accelerator_id = Some(clean_scalar(value));
            }
            "accelerator" if indent == 2 && key == "confidence" => {
                accelerator_confidence = Some(clean_scalar(value));
            }
            "accelerator" if indent == 2 && key == "scratchpad_bytes" => {
                accelerator_scratchpad_bytes = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "tensor_alignment_bytes" => {
                accelerator_tensor_alignment_bytes = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "dma_burst_bytes" => {
                accelerator_dma_burst_bytes = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "dma_setup_ns" => {
                accelerator_dma_setup_ns = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "dma_read_bytes_per_second" => {
                accelerator_dma_read_bytes_per_second = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "dma_write_bytes_per_second" => {
                accelerator_dma_write_bytes_per_second = parse_u64(value, key)?;
            }
            "accelerator" if indent == 2 && key == "spill_allowed" => {
                accelerator_spill_allowed = Some(parse_bool(value, key)?);
            }
            "recipes" => parse_recipe_line(
                &mut state,
                &mut replacement_recipes,
                indent,
                key,
                value,
            )?,
            "ops" => parse_ops_line(&mut state, &mut allowed_ops, indent, key, value)?,
            _ => {
                return Err(format!(
                    "unsupported profile field {}.{} at indentation {}",
                    state.section, key, indent
                ));
            }
        }
    }

    if profile_version != "edgefit.target.v1" {
        return Err("profile_version must be edgefit.target.v1".to_string());
    }

    if accelerator_section_present
        && accelerator_id.as_deref().is_none_or(str::is_empty)
    {
        return Err("accelerator.id is required when an accelerator section is declared".to_string());
    }
    let accelerator = match accelerator_id {
        Some(id) => Some(AcceleratorProfile {
            id,
            confidence: accelerator_confidence.unwrap_or_default(),
            scratchpad_bytes: accelerator_scratchpad_bytes.unwrap_or(0),
            tensor_alignment_bytes: accelerator_tensor_alignment_bytes.unwrap_or(0),
            dma_burst_bytes: accelerator_dma_burst_bytes.unwrap_or(0),
            dma_setup_ns: accelerator_dma_setup_ns.unwrap_or(0),
            dma_read_bytes_per_second: accelerator_dma_read_bytes_per_second.unwrap_or(0),
            dma_write_bytes_per_second: accelerator_dma_write_bytes_per_second.unwrap_or(0),
            spill_allowed: accelerator_spill_allowed.unwrap_or(false),
        }),
        None => None,
    };

    Ok(TargetProfile {
        source,
        fingerprint: profile_fingerprint(text),
        metadata,
        target_id,
        target_name,
        target_class,
        flash_bytes,
        ram_bytes,
        model_file_budget_bytes,
        peak_activation_budget_bytes,
        weights_residency,
        tensor_alignment_bytes,
        shape_max_rank,
        allow_unknown_dims,
        symbol_bounds,
        runtime_name,
        static_shapes_required,
        dynamic_allocation_allowed,
        external_memory_allowed,
        dtype_allowed,
        dtype_preferred,
        fp32_allowed,
        max_opset_versions,
        allowed_ops,
        accelerator,
        replacement_recipes,
        quantization_required,
        require_int8,
        min_quantized_weight_fraction,
        min_quantized_operator_coverage,
    })
}

/// 使用稳定 FNV-1a 指纹绑定快照与目标 profile 的原始内容，不把 metadata 当作内容校验替代品。
fn profile_fingerprint(text: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn parse_ops_line(
    state: &mut ParseState,
    allowed_ops: &mut BTreeMap<(String, String), OpRule>,
    indent: usize,
    key: &str,
    value: &str,
) -> EdgeFitResult<()> {
    match indent {
        2 if key == "allow" => state.ops_allow = true,
        4 if state.ops_allow => state.ops_domain = key.to_string(),
        6 if !state.ops_domain.is_empty() => {
            state.current_op = Some(key.to_string());
            state.op_rule_section = None;
            allowed_ops
                .entry((state.ops_domain.clone(), key.to_string()))
                .or_insert(OpRule {
                    dtypes: BTreeSet::new(),
                    input_dtypes: BTreeMap::new(),
                    output_dtypes: BTreeMap::new(),
                    attributes: BTreeMap::new(),
                    max_rank: None,
                    workspace_bytes: 0,
                    first_output_inplace_input_index: None,
                    cpu_cost: None,
                    npu_cost: None,
                });
        }
        8 => {
            let Some(op) = state.current_op.clone() else {
                return Err("operator property without operator name".to_string());
            };
            let Some(rule) = allowed_ops.get_mut(&(state.ops_domain.clone(), op)) else {
                return Err("operator rule was not initialized".to_string());
            };
            match key {
                "dtypes" => {
                    rule.dtypes = parse_list(value)
                        .into_iter()
                        .map(|item| edgefit_ir::normalize_dtype(&item))
                        .collect();
                }
                "max_rank" => rule.max_rank = parse_u64(value, key)?,
                "workspace_bytes" => {
                    rule.workspace_bytes = parse_u64(value, key)?.ok_or_else(|| {
                        "operator workspace_bytes requires an integer".to_string()
                    })?;
                }
                "first_output_inplace_input_index" => {
                    let input_index = parse_u64(value, key)?.ok_or_else(|| {
                        "operator first_output_inplace_input_index requires an integer".to_string()
                    })?;
                    rule.first_output_inplace_input_index = Some(
                        usize::try_from(input_index).map_err(|_| {
                            "operator first_output_inplace_input_index is too large".to_string()
                        })?,
                    );
                }
                "input_dtypes" | "output_dtypes" | "attributes" | "cpu_cost" | "npu_cost"
                    if value.is_empty() => {
                    state.op_rule_section = Some(key.to_string());
                    if key == "cpu_cost" {
                        rule.cpu_cost = Some(KernelCost::default());
                    } else if key == "npu_cost" {
                        rule.npu_cost = Some(KernelCost::default());
                    }
                }
                _ => return Err(format!("unsupported operator rule field {key}")),
            }
        }
        10 => {
            let Some(op) = state.current_op.clone() else {
                return Err("operator contract without operator name".to_string());
            };
            let Some(rule) = allowed_ops.get_mut(&(state.ops_domain.clone(), op)) else {
                return Err("operator rule was not initialized".to_string());
            };
            match state.op_rule_section.as_deref() {
                Some("input_dtypes") => {
                    insert_port_dtypes(&mut rule.input_dtypes, key, value, "input_dtypes")?;
                }
                Some("output_dtypes") => {
                    insert_port_dtypes(&mut rule.output_dtypes, key, value, "output_dtypes")?;
                }
                Some("attributes") => {
                    let allowed = parse_list(value)
                        .into_iter()
                        .map(|item| parse_attribute_contract_value(&item))
                        .collect::<EdgeFitResult<BTreeSet<_>>>()?;
                    if allowed.is_empty() {
                        return Err(format!(
                            "operator attribute {key} must contain typed values such as int:1 or string:NOTSET"
                        ));
                    }
                    if rule.attributes.insert(key.to_string(), allowed).is_some() {
                        return Err(format!("duplicate operator attribute constraint {key}"));
                    }
                }
                Some("cpu_cost") => {
                    parse_kernel_cost_field(rule.cpu_cost.as_mut(), key, value)?;
                }
                Some("npu_cost") => {
                    parse_kernel_cost_field(rule.npu_cost.as_mut(), key, value)?;
                }
                _ => return Err(format!(
                    "unsupported operator contract field {key} at indentation {indent}"
                )),
            }
        }
        _ => return Err(format!("unsupported ops field {key} at indentation {indent}")),
    }
    Ok(())
}

fn parse_kernel_cost_field(
    cost: Option<&mut KernelCost>,
    key: &str,
    value: &str,
) -> EdgeFitResult<()> {
    let cost = cost.ok_or("kernel cost section was not initialized")?;
    match key {
        "id" => cost.id = clean_scalar(value),
        "kind" => cost.kind = clean_scalar(value),
        "fixed_ns" => {
            cost.fixed_ns = parse_u64(value, key)?.ok_or("kernel fixed_ns requires an integer")?;
        }
        "throughput_per_second" => {
            cost.throughput_per_second = parse_u64(value, key)?
                .ok_or("kernel throughput_per_second requires an integer")?;
        }
        _ => return Err(format!("unsupported kernel cost field {key}")),
    }
    Ok(())
}

fn parse_recipe_line(
    state: &mut ParseState,
    recipes: &mut BTreeMap<(String, String), ReplacementRecipe>,
    indent: usize,
    key: &str,
    value: &str,
) -> EdgeFitResult<()> {
    match indent {
        2 => state.recipe_domain = key.to_string(),
        4 if !state.recipe_domain.is_empty() => {
            state.current_recipe_op = Some(key.to_string());
            recipes
                .entry((state.recipe_domain.clone(), key.to_string()))
                .or_default();
        }
        6 => {
            let op = state
                .current_recipe_op
                .clone()
                .ok_or("recipe field without operator")?;
            let recipe = recipes
                .get_mut(&(state.recipe_domain.clone(), op))
                .ok_or("recipe was not initialized")?;
            match key {
                "id" => recipe.id = clean_scalar(value),
                "trusted" => recipe.trusted = parse_bool(value, key)?,
                "source" => recipe.source = clean_scalar(value),
                "version" => recipe.version = clean_scalar(value),
                "replacement_ops" => recipe.replacement_ops = parse_list(value),
                _ => return Err(format!("unsupported recipe field {key}")),
            }
        }
        _ => return Err(format!("unsupported recipes field {key} at indentation {indent}")),
    }
    Ok(())
}

/// 属性值必须带类型前缀，避免整数 `1` 与字符串 `"1"` 被误判为同一能力。
fn parse_attribute_contract_value(value: &str) -> EdgeFitResult<AttributeValue> {
    let (kind, payload) = value
        .split_once(':')
        .ok_or_else(|| format!("operator attribute value {value} requires a type prefix"))?;
    if payload.is_empty() {
        return Err(format!("operator attribute value {value} requires a value"));
    }
    match kind {
        "int" => payload
            .parse::<i64>()
            .map(AttributeValue::Int)
            .map_err(|_| format!("operator attribute value {value} requires an int64")),
        "float" => {
            let parsed = payload
                .parse::<f64>()
                .map_err(|_| format!("operator attribute value {value} requires a float"))?;
            if !parsed.is_finite() {
                return Err(format!("operator attribute value {value} must be finite"));
            }
            Ok(AttributeValue::Float(FloatAttribute::from_f64(parsed)))
        }
        "string" => Ok(AttributeValue::String(payload.to_string())),
        "ints" => Ok(AttributeValue::Ints(parse_attribute_array(
            payload,
            value,
            |item| item.parse::<i64>().map_err(|_| ()),
        )?)),
        "floats" => {
            let values = parse_attribute_array(payload, value, |item| {
                item.parse::<f64>().map_err(|_| ()).and_then(|number| {
                    if number.is_finite() { Ok(number) } else { Err(()) }
                })
            })?;
            Ok(AttributeValue::Floats(
                values.into_iter().map(FloatAttribute::from_f64).collect(),
            ))
        }
        "strings" => Ok(AttributeValue::Strings(
            payload.split(';').map(str::to_string).collect(),
        )),
        _ => Err(format!("unsupported operator attribute value type {kind}")),
    }
}

fn parse_attribute_array<T>(
    payload: &str,
    original: &str,
    parse: impl Fn(&str) -> Result<T, ()>,
) -> EdgeFitResult<Vec<T>> {
    payload
        .split(';')
        .map(|item| {
            parse(item).map_err(|_| format!("invalid operator attribute array value {original}"))
        })
        .collect()
}

/// 严格解析端口键，避免 `00` 等多种文本映射到同一端口而形成歧义。
fn insert_port_dtypes(
    contracts: &mut BTreeMap<usize, BTreeSet<String>>,
    key: &str,
    value: &str,
    scope: &str,
) -> EdgeFitResult<()> {
    let port = key
        .parse::<usize>()
        .map_err(|_| format!("operator {scope} port {key} must be a zero-based integer"))?;
    if port.to_string() != key {
        return Err(format!(
            "operator {scope} port {key} must use canonical zero-based integer syntax"
        ));
    }
    let allowed = parse_list(value)
        .into_iter()
        .map(|item| edgefit_ir::normalize_dtype(&item))
        .collect::<BTreeSet<_>>();
    if allowed.is_empty() {
        return Err(format!(
            "operator {scope} port {port} must contain at least one dtype"
        ));
    }
    if contracts.insert(port, allowed).is_some() {
        return Err(format!("duplicate operator {scope} port {port}"));
    }
    Ok(())
}

fn parse_list(value: &str) -> Vec<String> {
    let value = value.trim();
    let Some(inner) = value
        .strip_prefix('[')
        .and_then(|item| item.strip_suffix(']'))
    else {
        return Vec::new();
    };
    inner
        .split(',')
        .map(clean_scalar)
        .filter(|item| !item.is_empty())
        .collect()
}

fn clean_scalar(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn parse_symbol_max(value: &str) -> EdgeFitResult<Option<u64>> {
    let value = value.trim();
    if value.is_empty() {
        return Err("shape symbol bound must provide a max value".to_string());
    }
    if let Some(inner) = value
        .strip_prefix('{')
        .and_then(|item| item.strip_suffix('}'))
    {
        for part in inner.split(',') {
            let Some((key, raw)) = part.split_once(':') else {
                return Err("shape symbol bound entries must use key: value".to_string());
            };
            if key.trim() == "max" {
                return raw
                    .trim()
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|err| format!("shape symbol max must be an integer: {err}"));
            }
            return Err(format!("unsupported shape symbol field {}", key.trim()));
        }
        return Err("shape symbol bound must contain max".to_string());
    }
    value
        .parse::<u64>()
        .map(Some)
        .map_err(|err| format!("shape symbol max must be an integer: {err}"))
}

fn parse_u64(value: &str, key: &str) -> EdgeFitResult<Option<u64>> {
    if value.trim().is_empty() {
        return Ok(None);
    }
    value
        .trim()
        .parse::<u64>()
        .map(Some)
        .map_err(|err| format!("{key} must be an integer: {err}"))
}

fn parse_bool(value: &str, key: &str) -> EdgeFitResult<bool> {
    match value.trim() {
        "true" | "True" => Ok(true),
        "false" | "False" => Ok(false),
        _ => Err(format!("{key} must be true or false")),
    }
}

fn parse_f64(value: &str, key: &str) -> EdgeFitResult<f64> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|err| format!("{key} must be a number: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_core_profile_fields() {
        let text = r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-09
target:
  id: demo
memory:
  model_file_budget_bytes: 10
  peak_activation_budget_bytes: 20
  tensor_alignment_bytes: 16
shape:
  max_rank: 4
  allow_unknown_dims: false
  symbols:
    N: { max: 1 }
runtime:
  static_shapes_required: true
dtype:
  allowed: [int8]
ops:
  allow:
    ai.onnx:
      Conv:
        dtypes: [int8]
        input_dtypes:
          0: [int8]
          1: [int32, int64]
        output_dtypes:
          0: [int8]
        attributes:
          auto_pad: [string:NOTSET, string:SAME_UPPER]
          group: [int:1, int:2]
        workspace_bytes: 8
        first_output_inplace_input_index: 0
    com.microsoft:
      QLinearAdd:
        dtypes: [uint8]
quantization:
  required: true
  min_quantized_weight_fraction: 0.95
"#;
        let profile = parse_profile(text, PathBuf::from("target.yaml")).unwrap();
        assert_eq!(profile.target_id, "demo");
        assert_eq!(profile.metadata.confidence, "seed");
        assert!(profile.static_shapes_required);
        assert!(profile.op_rule("ai.onnx", "Conv").is_some());
        assert!(profile.op_rule("com.microsoft", "QLinearAdd").is_some());
        assert_eq!(profile.model_file_budget_bytes, Some(10));
        assert_eq!(profile.tensor_alignment_bytes, 16);
        assert_eq!(profile.shape_max_rank, Some(4));
        assert!(!profile.allow_unknown_dims);
        assert_eq!(profile.symbol_bounds.get("N"), Some(&1));
        let conv = profile.op_rule("ai.onnx", "Conv").unwrap();
        assert_eq!(conv.workspace_bytes, 8);
        assert_eq!(conv.first_output_inplace_input_index, Some(0));
        assert_eq!(conv.input_dtypes.get(&0).unwrap(), &BTreeSet::from(["int8".to_string()]));
        assert_eq!(
            conv.input_dtypes.get(&1).unwrap(),
            &BTreeSet::from(["int32".to_string(), "int64".to_string()])
        );
        assert_eq!(conv.output_dtypes.get(&0).unwrap(), &BTreeSet::from(["int8".to_string()]));
        assert_eq!(
            conv.attributes.get("auto_pad").unwrap(),
            &BTreeSet::from([
                AttributeValue::String("NOTSET".to_string()),
                AttributeValue::String("SAME_UPPER".to_string()),
            ])
        );
    }

    #[test]
    fn defaults_memory_planner_fields_for_legacy_profile() {
        let text = r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-10
target:
  id: demo
ops:
  allow:
    ai.onnx:
      Relu:
        dtypes: [int8]
"#;
        let profile = parse_profile(text, PathBuf::from("target.yaml")).unwrap();
        let relu = profile.op_rule("ai.onnx", "Relu").unwrap();

        assert_eq!(profile.tensor_alignment_bytes, 1);
        assert_eq!(relu.workspace_bytes, 0);
        assert_eq!(relu.first_output_inplace_input_index, None);
        assert!(relu.input_dtypes.is_empty());
        assert!(relu.output_dtypes.is_empty());
        assert!(relu.attributes.is_empty());
    }

    #[test]
    fn rejects_invalid_or_duplicate_operator_contract_ports() {
        for contracts in [
            "input_dtypes:\n          first: [int8]",
            "input_dtypes:\n          00: [int8]",
            "input_dtypes:\n          0: []",
            "input_dtypes:\n          0: [int8]\n          0: [uint8]",
        ] {
            let text = format!(
                r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-11
target:
  id: demo
ops:
  allow:
    ai.onnx:
      Add:
        dtypes: [int8]
        {contracts}
"#
            );

            assert!(parse_profile(&text, PathBuf::from("target.yaml")).is_err());
        }
    }

    #[test]
    fn rejects_empty_or_duplicate_operator_attribute_constraints() {
        for contracts in [
            "attributes:\n          axis: []",
            "attributes:\n          axis: [int:0]\n          axis: [int:1]",
            "attributes:\n          axis: [0]",
        ] {
            let text = format!(
                r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-11
target:
  id: demo
ops:
  allow:
    ai.onnx:
      Softmax:
        dtypes: [float32]
        {contracts}
"#
            );

            assert!(parse_profile(&text, PathBuf::from("target.yaml")).is_err());
        }
    }

    #[test]
    fn rejects_invalid_tensor_alignment() {
        for alignment in [0, 3] {
            let text = format!(
                r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-10
target:
  id: demo
memory:
  tensor_alignment_bytes: {alignment}
ops:
  allow:
    ai.onnx:
      Relu:
        dtypes: [int8]
"#
            );
            let profile = parse_profile(&text, PathBuf::from("target.yaml")).unwrap();
            let error = profile.validate().unwrap_err();

            assert!(error.contains("memory.tensor_alignment_bytes"));
        }
    }

    #[test]
    fn rejects_operator_workspace_larger_than_ram() {
        let text = r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-10
target:
  id: demo
memory:
  ram_bytes: 1024
ops:
  allow:
    ai.onnx:
      Conv:
        dtypes: [int8]
        workspace_bytes: 1025
"#;
        let profile = parse_profile(text, PathBuf::from("target.yaml")).unwrap();
        let error = profile.validate().unwrap_err();

        assert!(error.contains("workspace_bytes cannot exceed memory.ram_bytes"));
    }

    #[test]
    fn rejects_profile_without_metadata() {
        let text = r#"
profile_version: edgefit.target.v1
target:
  id: demo
dtype:
  allowed: [int8]
ops:
  allow:
    ai.onnx:
      Conv:
        dtypes: [int8]
"#;
        let profile = parse_profile(text, PathBuf::from("target.yaml")).unwrap();
        let error = profile.validate().unwrap_err();
        assert!(error.contains("metadata.source"));
    }

    #[test]
    fn rejects_operator_rule_without_dtype_scope() {
        let text = r#"
profile_version: edgefit.target.v1
metadata:
  source: test profile
  confidence: seed
  last_verified: 2026-07-09
target:
  id: demo
dtype:
  allowed: [int8]
ops:
  allow:
    ai.onnx:
      Conv:
"#;
        let profile = parse_profile(text, PathBuf::from("target.yaml")).unwrap();
        let error = profile.validate().unwrap_err();
        assert!(error.contains("dtypes"));
    }

    #[test]
    fn rejects_partial_accelerator_without_id() {
        let text = include_str!("../../../targets/virtual-npu.yaml")
            .replace("id: generic-npu-v1", "id:");

        let error = parse_profile(&text, PathBuf::from("target.yaml")).unwrap_err();

        assert!(error.contains("accelerator.id is required"));
    }

    #[test]
    fn parses_virtual_accelerator_costs_and_trusted_recipes() {
        let profile = parse_profile(
            include_str!("../../../targets/virtual-npu.yaml"),
            PathBuf::from("targets/virtual-npu.yaml"),
        )
        .unwrap();

        profile.validate().unwrap();
        assert_eq!(profile.accelerator.as_ref().unwrap().dma_burst_bytes, 64);
        assert_eq!(
            profile
                .op_rule("ai.onnx", "Add")
                .unwrap()
                .npu_cost
                .as_ref()
                .unwrap()
                .id,
            "npu.add.int8"
        );
        assert_eq!(
            profile
                .replacement_recipes
                .get(&("ai.onnx".to_string(), "HardSwish".to_string()))
                .unwrap()
                .replacement_ops,
            ["HardSigmoid", "Mul"]
        );
    }
}
