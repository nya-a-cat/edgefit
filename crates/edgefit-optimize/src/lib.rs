//! 硬件感知的确定性优化计划器。
//!
//! 本模块只生成可审计计划，不改写 ONNX，也不把 seed 代价冒充真实硬件测量。

use edgefit_ir::{escape_json, AttributeValue, Dim, EdgeFitResult, NodeInfo, NormalizedModel};
use edgefit_target::{KernelCost, OpRule, ReplacementRecipe, TargetProfile};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeAssignment {
    pub node_index: usize,
    pub op_type: String,
    pub device: String,
    pub kernel_id: Option<String>,
    pub recipe_id: Option<String>,
    pub launch_ns: Option<u64>,
    pub compute_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferEvent {
    pub kind: String,
    pub tensor: String,
    pub bytes: u64,
    pub at_node: usize,
    pub latency_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub id: usize,
    pub first_node: usize,
    pub last_node: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizationPlan {
    pub schema: String,
    pub status: String,
    pub model_sha256: String,
    pub target_id: String,
    pub target_fingerprint: String,
    pub accelerator_id: String,
    pub confidence: String,
    pub baseline_blockers: u64,
    pub proposed_blockers: u64,
    pub baseline_latency_ns: Option<u64>,
    pub proposed_latency_ns: Option<u64>,
    pub launch_latency_ns: Option<u64>,
    pub compute_latency_ns: Option<u64>,
    pub transfer_latency_ns: u64,
    pub transfer_bytes: u64,
    pub spill_bytes: u64,
    pub peak_scratchpad_bytes: u64,
    pub assignments: Vec<NodeAssignment>,
    pub segments: Vec<Segment>,
    pub events: Vec<TransferEvent>,
    pub blockers: Vec<String>,
    pub plan_hash: String,
}

pub fn optimize(model: &NormalizedModel, profile: &TargetProfile) -> EdgeFitResult<OptimizationPlan> {
    profile.validate()?;
    let accelerator = profile
        .accelerator
        .as_ref()
        .ok_or("target profile does not declare an accelerator")?;
    let producers = producers(model);
    let graph_outputs = model.outputs.iter().cloned().collect::<BTreeSet<_>>();
    let mut assignments: Vec<NodeAssignment> = Vec::with_capacity(model.nodes.len());
    let mut baseline_blockers = 0_u64;
    let mut baseline_latency = Some(0_u64);

    for (node_index, node) in model.nodes.iter().enumerate() {
        let rule = profile.op_rule(&node.domain, &node.op_type);
        let cpu = rule
            .filter(|rule| contract_compatible(node, rule, model))
            .and_then(|rule| rule.cpu_cost.as_ref())
            .and_then(|cost| evaluate_cost(cost, node, model));
        baseline_latency = add_optional_latency(baseline_latency, cpu.map(|item| item.0 + item.1));
        if cpu.is_none() {
            baseline_blockers += 1;
        }

        let direct = rule
            .filter(|rule| contract_compatible(node, rule, model))
            .and_then(|rule| rule.npu_cost.as_ref())
            .and_then(|cost| evaluate_cost(cost, node, model).map(|latency| (cost, latency, None)));
        let recipe = profile
            .replacement_recipes
            .get(&(node.domain.clone(), node.op_type.clone()))
            .and_then(|recipe| recipe_candidate(recipe, node, model, profile));
        let npu = direct.or(recipe);

        let choose_npu = match (cpu, &npu) {
            (None, Some(_)) => true,
            (Some((cpu_launch, cpu_compute)), Some((_, (npu_launch, npu_compute), _))) => {
                let boundary_bytes = node
                    .inputs
                    .iter()
                    .filter(|name| !name.is_empty())
                    .filter(|name| {
                        producers
                            .get(*name)
                            .and_then(|producer| assignments.get(*producer))
                            .is_none_or(|assignment| assignment.device != "npu")
                    })
                    .chain(node.outputs.iter().filter(|name| graph_outputs.contains(*name)))
                    .filter_map(|name| tensor_bytes(model, name))
                    .map(|bytes| align_up(bytes, accelerator.dma_burst_bytes))
                    .sum::<u64>();
                let boundary_ns = dma_ns(
                    boundary_bytes,
                    accelerator.dma_setup_ns,
                    accelerator.dma_read_bytes_per_second,
                );
                npu_launch
                    .saturating_add(*npu_compute)
                    .saturating_add(boundary_ns)
                    < cpu_launch.saturating_add(cpu_compute)
            }
            _ => false,
        };

        if choose_npu {
            let (cost, (launch, compute), recipe_id) = npu.expect("checked NPU candidate");
            assignments.push(NodeAssignment {
                node_index,
                op_type: node.op_type.clone(),
                device: "npu".to_string(),
                kernel_id: Some(cost.id.clone()),
                recipe_id,
                launch_ns: Some(launch),
                compute_ns: Some(compute),
            });
        } else if let Some(rule) = rule {
            let latency = rule.cpu_cost.as_ref().and_then(|cost| evaluate_cost(cost, node, model));
            assignments.push(NodeAssignment {
                node_index,
                op_type: node.op_type.clone(),
                device: if latency.is_some() { "cpu" } else { "unsupported" }.to_string(),
                kernel_id: rule.cpu_cost.as_ref().map(|cost| cost.id.clone()),
                recipe_id: None,
                launch_ns: latency.map(|item| item.0),
                compute_ns: latency.map(|item| item.1),
            });
        } else {
            assignments.push(NodeAssignment {
                node_index,
                op_type: node.op_type.clone(),
                device: "unsupported".to_string(),
                kernel_id: None,
                recipe_id: None,
                launch_ns: None,
                compute_ns: None,
            });
        }
    }

    let segments = collect_segments(&assignments);
    let simulation = simulate_npu(model, profile, &segments)?;
    let proposed_blockers = assignments
        .iter()
        .filter(|assignment| assignment.device == "unsupported")
        .count() as u64
        + simulation.blockers.len() as u64;
    let launch_latency_ns = sum_optional(assignments.iter().map(|item| item.launch_ns));
    let compute_latency_ns = sum_optional(assignments.iter().map(|item| item.compute_ns));
    let proposed_latency_ns = launch_latency_ns
        .zip(compute_latency_ns)
        .map(|(launch, compute)| launch.saturating_add(compute).saturating_add(simulation.transfer_ns));
    let mut blockers = assignments
        .iter()
        .filter(|assignment| assignment.device == "unsupported")
        .map(|assignment| format!("node:{} unsupported", assignment.node_index))
        .collect::<Vec<_>>();
    blockers.extend(simulation.blockers);
    if proposed_latency_ns.is_none() {
        blockers.push("latency_unknown".to_string());
    }
    let status = if blockers.is_empty() { "pass" } else { "fail" }.to_string();
    let plan_hash = plan_fingerprint(
        &model.sha256,
        &profile.fingerprint,
        &assignments,
        &segments,
        &simulation.events,
    );

    Ok(OptimizationPlan {
        schema: "edgefit.optimization_plan.v1".to_string(),
        status,
        model_sha256: model.sha256.clone(),
        target_id: profile.target_id.clone(),
        target_fingerprint: profile.fingerprint.clone(),
        accelerator_id: accelerator.id.clone(),
        confidence: accelerator.confidence.clone(),
        baseline_blockers,
        proposed_blockers,
        baseline_latency_ns: baseline_latency,
        proposed_latency_ns,
        launch_latency_ns,
        compute_latency_ns,
        transfer_latency_ns: simulation.transfer_ns,
        transfer_bytes: simulation.transfer_bytes,
        spill_bytes: simulation.spill_bytes,
        peak_scratchpad_bytes: simulation.peak_bytes,
        assignments,
        segments,
        events: simulation.events,
        blockers,
        plan_hash,
    })
}

struct Simulation {
    events: Vec<TransferEvent>,
    transfer_ns: u64,
    transfer_bytes: u64,
    spill_bytes: u64,
    peak_bytes: u64,
    blockers: Vec<String>,
}

fn simulate_npu(
    model: &NormalizedModel,
    profile: &TargetProfile,
    segments: &[Segment],
) -> EdgeFitResult<Simulation> {
    let accelerator = profile.accelerator.as_ref().ok_or("missing accelerator")?;
    let consumers = consumers(model);
    let graph_outputs = model.outputs.iter().cloned().collect::<BTreeSet<_>>();
    let mut events = Vec::new();
    let mut transfer_ns = 0_u64;
    let mut transfer_bytes = 0_u64;
    let mut spill_bytes = 0_u64;
    let mut peak_bytes = 0_u64;
    let mut blockers = Vec::new();

    for segment in segments {
        let mut resident = BTreeMap::<String, u64>::new();
        let mut spilled = BTreeSet::<String>::new();
        let mut used = 0_u64;
        for node_index in segment.first_node..=segment.last_node {
            let node = &model.nodes[node_index];
            let protected = node.inputs.iter().cloned().collect::<BTreeSet<_>>();
            for input in node.inputs.iter().filter(|name| !name.is_empty()) {
                if resident.contains_key(input) {
                    continue;
                }
                let Some(bytes) = tensor_bytes(model, input) else {
                    blockers.push(format!("node:{node_index} tensor:{input} size_unknown"));
                    continue;
                };
                let bytes = align_up(bytes, accelerator.tensor_alignment_bytes);
                spill_until_fit(
                    node_index,
                    bytes,
                    &protected,
                    &consumers,
                    &mut resident,
                    &mut spilled,
                    &mut used,
                    accelerator,
                    &mut events,
                    &mut transfer_ns,
                    &mut transfer_bytes,
                    &mut spill_bytes,
                    &mut blockers,
                );
                if used.saturating_add(bytes) <= accelerator.scratchpad_bytes {
                    resident.insert(input.clone(), bytes);
                    used = used.saturating_add(bytes);
                    record_transfer(
                        if spilled.remove(input) { "reload" } else { "load" },
                        input,
                        bytes,
                        node_index,
                        accelerator.dma_read_bytes_per_second,
                        accelerator,
                        &mut events,
                        &mut transfer_ns,
                        &mut transfer_bytes,
                    );
                } else {
                    blockers.push(format!("node:{node_index} tensor:{input} scratchpad_unavailable"));
                }
            }
            for output in node.outputs.iter().filter(|name| !name.is_empty()) {
                let Some(bytes) = tensor_bytes(model, output) else {
                    blockers.push(format!("node:{node_index} tensor:{output} size_unknown"));
                    continue;
                };
                let bytes = align_up(bytes, accelerator.tensor_alignment_bytes);
                spill_until_fit(
                    node_index,
                    bytes,
                    &protected,
                    &consumers,
                    &mut resident,
                    &mut spilled,
                    &mut used,
                    accelerator,
                    &mut events,
                    &mut transfer_ns,
                    &mut transfer_bytes,
                    &mut spill_bytes,
                    &mut blockers,
                );
                if used.saturating_add(bytes) <= accelerator.scratchpad_bytes {
                    resident.insert(output.clone(), bytes);
                    used = used.saturating_add(bytes);
                }
            }
            peak_bytes = peak_bytes.max(used);
            resident.retain(|tensor, bytes| {
                let keep = consumers
                    .get(tensor)
                    .is_some_and(|uses| uses.iter().any(|index| *index > node_index && *index <= segment.last_node))
                    || graph_outputs.contains(tensor)
                    || consumers
                        .get(tensor)
                        .is_some_and(|uses| uses.iter().any(|index| *index > segment.last_node));
                if !keep {
                    used = used.saturating_sub(*bytes);
                }
                keep
            });
        }
        let last_node = segment.last_node;
        for (tensor, bytes) in resident {
            record_transfer(
                "store",
                &tensor,
                bytes,
                last_node,
                accelerator.dma_write_bytes_per_second,
                accelerator,
                &mut events,
                &mut transfer_ns,
                &mut transfer_bytes,
            );
        }
    }

    Ok(Simulation { events, transfer_ns, transfer_bytes, spill_bytes, peak_bytes, blockers })
}

#[allow(clippy::too_many_arguments)]
fn spill_until_fit(
    node_index: usize,
    required: u64,
    protected: &BTreeSet<String>,
    consumers: &BTreeMap<String, Vec<usize>>,
    resident: &mut BTreeMap<String, u64>,
    spilled: &mut BTreeSet<String>,
    used: &mut u64,
    accelerator: &edgefit_target::AcceleratorProfile,
    events: &mut Vec<TransferEvent>,
    transfer_ns: &mut u64,
    transfer_bytes: &mut u64,
    spill_bytes: &mut u64,
    blockers: &mut Vec<String>,
) {
    while used.saturating_add(required) > accelerator.scratchpad_bytes {
        if !accelerator.spill_allowed {
            blockers.push(format!("node:{node_index} scratchpad_exceeded"));
            return;
        }
        let victim = resident
            .iter()
            .filter(|(tensor, _)| !protected.contains(*tensor))
            .max_by_key(|(tensor, bytes)| {
                let next = consumers
                    .get(*tensor)
                    .and_then(|uses| uses.iter().find(|index| **index > node_index))
                    .copied()
                    .unwrap_or(usize::MAX);
                (next, **bytes, (*tensor).clone())
            })
            .map(|(tensor, bytes)| (tensor.clone(), *bytes));
        let Some((tensor, bytes)) = victim else {
            blockers.push(format!("node:{node_index} no_spill_victim"));
            return;
        };
        resident.remove(&tensor);
        spilled.insert(tensor.clone());
        *used = used.saturating_sub(bytes);
        *spill_bytes = spill_bytes.saturating_add(bytes);
        record_transfer(
            "spill",
            &tensor,
            bytes,
            node_index,
            accelerator.dma_write_bytes_per_second,
            accelerator,
            events,
            transfer_ns,
            transfer_bytes,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn record_transfer(
    kind: &str,
    tensor: &str,
    bytes: u64,
    at_node: usize,
    bandwidth: u64,
    accelerator: &edgefit_target::AcceleratorProfile,
    events: &mut Vec<TransferEvent>,
    transfer_ns: &mut u64,
    transfer_bytes: &mut u64,
) {
    let bytes = align_up(bytes, accelerator.dma_burst_bytes);
    let latency_ns = dma_ns(bytes, accelerator.dma_setup_ns, bandwidth);
    *transfer_ns = transfer_ns.saturating_add(latency_ns);
    *transfer_bytes = transfer_bytes.saturating_add(bytes);
    events.push(TransferEvent {
        kind: kind.to_string(),
        tensor: tensor.to_string(),
        bytes,
        at_node,
        latency_ns,
    });
}

fn recipe_candidate<'a>(
    recipe: &'a ReplacementRecipe,
    node: &NodeInfo,
    model: &NormalizedModel,
    profile: &'a TargetProfile,
) -> Option<(&'a KernelCost, (u64, u64), Option<String>)> {
    if !recipe.trusted {
        return None;
    }
    let mut first = None;
    let mut launch = 0_u64;
    let mut compute = 0_u64;
    for replacement in &recipe.replacement_ops {
        let cost = profile.op_rule(&node.domain, replacement)?.npu_cost.as_ref()?;
        let latency = evaluate_cost(cost, node, model)?;
        first.get_or_insert(cost);
        launch = launch.saturating_add(latency.0);
        compute = compute.saturating_add(latency.1);
    }
    Some((first?, (launch, compute), Some(recipe.id.clone())))
}

fn evaluate_cost(cost: &KernelCost, node: &NodeInfo, model: &NormalizedModel) -> Option<(u64, u64)> {
    let units = match cost.kind.as_str() {
        "fixed" => 0,
        "element" => node.outputs.first().and_then(|name| tensor_elements(model, name))?,
        "bytes" => node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .map(|name| tensor_bytes(model, name))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .sum(),
        "mac" => mac_units(node, model)?,
        _ => return None,
    };
    let compute = if units == 0 {
        0
    } else {
        ceil_mul_div(units, 1_000_000_000, cost.throughput_per_second)?
    };
    Some((cost.fixed_ns, compute))
}

fn mac_units(node: &NodeInfo, model: &NormalizedModel) -> Option<u64> {
    match node.op_type.as_str() {
        "MatMul" | "Gemm" => {
            let left = model.tensors.get(node.inputs.first()?)?.shape.as_ref()?;
            let right = model.tensors.get(node.inputs.get(1)?)?.shape.as_ref()?;
            let m = known_dim(left.get(left.len().checked_sub(2)?)?)?;
            let k = known_dim(left.last()?)?;
            let n = known_dim(right.last()?)?;
            m.checked_mul(k)?.checked_mul(n)
        }
        "Conv" => {
            let output_elements = tensor_elements(model, node.outputs.first()?)?;
            let input = model.tensors.get(node.inputs.first()?)?.shape.as_ref()?;
            let channels = known_dim(input.get(1)?)?;
            let group = match node.attributes.get("group") {
                Some(AttributeValue::Int(value)) if *value > 0 => *value as u64,
                _ => 1,
            };
            let kernel = match node.attributes.get("kernel_shape") {
                Some(AttributeValue::Ints(values)) => values.iter().try_fold(1_u64, |total, value| {
                    if *value > 0 { total.checked_mul(*value as u64) } else { None }
                })?,
                _ => 1,
            };
            output_elements.checked_mul(channels / group)?.checked_mul(kernel)
        }
        _ => None,
    }
}

fn contract_compatible(node: &NodeInfo, rule: &OpRule, model: &NormalizedModel) -> bool {
    if node
        .inputs
        .iter()
        .chain(node.outputs.iter())
        .filter(|name| !name.is_empty())
        .any(|name| {
            model
                .tensors
                .get(name)
                .and_then(|tensor| tensor.dtype.as_ref())
                .is_none_or(|dtype| !rule.dtypes.contains(dtype))
        })
    {
        return false;
    }
    for (name, allowed) in &rule.attributes {
        if !node.attributes.get(name).is_some_and(|value| allowed.contains(value)) {
            return false;
        }
    }
    for (ports, names) in [(&rule.input_dtypes, &node.inputs), (&rule.output_dtypes, &node.outputs)] {
        for (port, allowed) in ports {
            let Some(dtype) = names
                .get(*port)
                .and_then(|name| model.tensors.get(name))
                .and_then(|tensor| tensor.dtype.as_ref())
            else {
                return false;
            };
            if !allowed.contains(dtype) {
                return false;
            }
        }
    }
    true
}

fn collect_segments(assignments: &[NodeAssignment]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut start = None;
    for (index, assignment) in assignments.iter().enumerate() {
        if assignment.device == "npu" {
            start.get_or_insert(index);
        } else if let Some(first) = start.take() {
            segments.push(Segment { id: segments.len(), first_node: first, last_node: index - 1 });
        }
    }
    if let Some(first) = start {
        segments.push(Segment { id: segments.len(), first_node: first, last_node: assignments.len() - 1 });
    }
    segments
}

fn producers(model: &NormalizedModel) -> BTreeMap<String, usize> {
    model.nodes.iter().enumerate().flat_map(|(index, node)| {
        node.outputs.iter().filter(|name| !name.is_empty()).map(move |name| (name.clone(), index))
    }).collect()
}

fn consumers(model: &NormalizedModel) -> BTreeMap<String, Vec<usize>> {
    let mut result = BTreeMap::<String, Vec<usize>>::new();
    for (index, node) in model.nodes.iter().enumerate() {
        for input in node.inputs.iter().filter(|name| !name.is_empty()) {
            result.entry(input.clone()).or_default().push(index);
        }
    }
    result
}

fn tensor_bytes(model: &NormalizedModel, name: &str) -> Option<u64> {
    model.tensors.get(name)?.byte_size_with_bounds(&BTreeMap::new()).map(|item| item.0)
}

fn tensor_elements(model: &NormalizedModel, name: &str) -> Option<u64> {
    let tensor = model.tensors.get(name)?;
    tensor.shape.as_ref()?.iter().try_fold(1_u64, |total, dim| {
        total.checked_mul(known_dim(dim)?)
    })
}

fn known_dim(dim: &Dim) -> Option<u64> {
    match dim { Dim::Known(value) if *value >= 0 => Some(*value as u64), _ => None }
}

fn add_optional_latency(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    current.zip(next).map(|(left, right)| left.saturating_add(right))
}

fn sum_optional(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    values.fold(Some(0_u64), add_optional_latency)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    let remainder = value % alignment;
    if remainder == 0 { value } else { value.saturating_add(alignment - remainder) }
}

fn dma_ns(bytes: u64, setup_ns: u64, bandwidth: u64) -> u64 {
    setup_ns.saturating_add(ceil_mul_div(bytes, 1_000_000_000, bandwidth).unwrap_or(u64::MAX))
}

fn ceil_mul_div(value: u64, multiplier: u64, divisor: u64) -> Option<u64> {
    if divisor == 0 { return None; }
    let numerator = u128::from(value).checked_mul(u128::from(multiplier))?;
    let result = numerator.checked_add(u128::from(divisor - 1))? / u128::from(divisor);
    u64::try_from(result).ok()
}

fn plan_fingerprint(
    model_sha256: &str,
    target_fingerprint: &str,
    assignments: &[NodeAssignment],
    segments: &[Segment],
    events: &[TransferEvent],
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let text = format!(
        "model={model_sha256};target={target_fingerprint};{assignments:?}{segments:?}{events:?}"
    );
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

pub fn render_plan(plan: &OptimizationPlan, format: &str) -> String {
    if format == "markdown" {
        return render_markdown(plan);
    }
    render_json(plan)
}

fn render_json(plan: &OptimizationPlan) -> String {
    let optional = |value: Option<u64>| value.map(|item| item.to_string()).unwrap_or_else(|| "null".to_string());
    let assignments = plan.assignments.iter().map(|item| format!(
        "{{\"node_index\":{},\"op_type\":\"{}\",\"device\":\"{}\",\"kernel_id\":{},\"recipe_id\":{},\"launch_ns\":{},\"compute_ns\":{}}}",
        item.node_index, escape_json(&item.op_type), item.device,
        json_optional_string(item.kernel_id.as_deref()), json_optional_string(item.recipe_id.as_deref()),
        optional(item.launch_ns), optional(item.compute_ns)
    )).collect::<Vec<_>>().join(",");
    let segments = plan.segments.iter().map(|item| format!(
        "{{\"id\":{},\"first_node\":{},\"last_node\":{}}}", item.id, item.first_node, item.last_node
    )).collect::<Vec<_>>().join(",");
    let events = plan.events.iter().map(|item| format!(
        "{{\"kind\":\"{}\",\"tensor\":\"{}\",\"bytes\":{},\"at_node\":{},\"latency_ns\":{}}}",
        item.kind, escape_json(&item.tensor), item.bytes, item.at_node, item.latency_ns
    )).collect::<Vec<_>>().join(",");
    let blockers = plan.blockers.iter().map(|item| format!("\"{}\"", escape_json(item))).collect::<Vec<_>>().join(",");
    format!(
        "{{\n  \"schema\": \"{}\",\n  \"status\": \"{}\",\n  \"model_sha256\": \"{}\",\n  \"target_id\": \"{}\",\n  \"target_fingerprint\": \"{}\",\n  \"accelerator_id\": \"{}\",\n  \"confidence\": \"{}\",\n  \"baseline\": {{\"blockers\":{},\"latency_ns\":{}}},\n  \"proposed\": {{\"blockers\":{},\"latency_ns\":{},\"launch_ns\":{},\"compute_ns\":{},\"transfer_ns\":{},\"transfer_bytes\":{},\"spill_bytes\":{},\"peak_scratchpad_bytes\":{}}},\n  \"assignments\": [{}],\n  \"segments\": [{}],\n  \"events\": [{}],\n  \"blockers\": [{}],\n  \"plan_hash\": \"{}\"\n}}\n",
        plan.schema, plan.status, escape_json(&plan.model_sha256), escape_json(&plan.target_id),
        escape_json(&plan.target_fingerprint), escape_json(&plan.accelerator_id), escape_json(&plan.confidence),
        plan.baseline_blockers, optional(plan.baseline_latency_ns), plan.proposed_blockers,
        optional(plan.proposed_latency_ns), optional(plan.launch_latency_ns), optional(plan.compute_latency_ns),
        plan.transfer_latency_ns, plan.transfer_bytes, plan.spill_bytes, plan.peak_scratchpad_bytes,
        assignments, segments, events, blockers, plan.plan_hash
    )
}

fn render_markdown(plan: &OptimizationPlan) -> String {
    format!(
        "# EdgeFit Optimization Plan\n\n**Status:** `{}`  \n**Accelerator:** `{}`  \n**Confidence:** `{}`  \n**Plan hash:** `{}`\n\n| Metric | Baseline | Proposed |\n| --- | ---: | ---: |\n| Blockers | {} | {} |\n| Latency (ns) | {} | {} |\n| Transfer bytes | - | {} |\n| Spill bytes | - | {} |\n| Peak scratchpad bytes | - | {} |\n\nNPU segments: {}. Remaining blockers: {}.\n",
        plan.status, plan.accelerator_id, plan.confidence, plan.plan_hash,
        plan.baseline_blockers, plan.proposed_blockers,
        display_optional(plan.baseline_latency_ns), display_optional(plan.proposed_latency_ns),
        plan.transfer_bytes, plan.spill_bytes, plan.peak_scratchpad_bytes,
        plan.segments.len(), if plan.blockers.is_empty() { "none".to_string() } else { plan.blockers.join(", ") }
    )
}

fn display_optional(value: Option<u64>) -> String {
    value.map(|item| item.to_string()).unwrap_or_else(|| "unknown".to_string())
}

fn json_optional_string(value: Option<&str>) -> String {
    value.map(|item| format!("\"{}\"", escape_json(item))).unwrap_or_else(|| "null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefit_ir::parse_normalized_model;
    use edgefit_target::parse_profile;
    use std::path::PathBuf;

    #[test]
    fn creates_a_deterministic_contiguous_npu_plan() {
        let model = parse_normalized_model(include_str!(
            "../../../examples/models/virtual_npu_tiny.edgefit.json"
        ))
        .unwrap();
        let profile = parse_profile(
            include_str!("../../../targets/virtual-npu.yaml"),
            PathBuf::from("targets/virtual-npu.yaml"),
        )
        .unwrap();

        let first = optimize(&model, &profile).unwrap();
        let second = optimize(&model, &profile).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.status, "pass");
        assert_eq!(first.segments.len(), 1);
        assert!(first.assignments.iter().all(|item| item.device == "npu"));
        assert!(first.events.iter().any(|item| item.kind == "load"));
        assert!(first.events.iter().any(|item| item.kind == "store"));
        assert!(first.proposed_latency_ns < first.baseline_latency_ns);
        let mut changed_profile = profile.clone();
        changed_profile.fingerprint = "fnv1a64:changed-profile".to_string();
        let changed_target = optimize(&model, &changed_profile).unwrap();
        assert_ne!(first.plan_hash, changed_target.plan_hash);
        let mut changed_model = model.clone();
        changed_model.sha256 = "sha256:changed-model".to_string();
        let changed_input = optimize(&changed_model, &profile).unwrap();
        assert_ne!(first.plan_hash, changed_input.plan_hash);
    }

    #[test]
    fn splits_npu_segments_around_a_cpu_assignment() {
        let model = parse_model(&[
            ("Relu", &["x"], &["a"]),
            ("Add", &["a", "bias"], &["b"]),
            ("Relu", &["b"], &["y"]),
        ]);
        let profile = parse_test_profile(262_144, true, 10_000, 1, 1);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "pass");
        assert_eq!(
            plan.assignments.iter().map(|item| item.device.as_str()).collect::<Vec<_>>(),
            vec!["npu", "cpu", "npu"]
        );
        assert_eq!(plan.segments.len(), 2);
        assert_eq!((plan.segments[0].first_node, plan.segments[0].last_node), (0, 0));
        assert_eq!((plan.segments[1].first_node, plan.segments[1].last_node), (2, 2));
        assert_eq!(plan.assignments[1].device, "cpu");
    }

    #[test]
    fn records_deterministic_spill_and_reload_events() {
        let model = parse_model(&[
            ("Relu", &["x"], &["a"]),
            ("Relu", &["bias"], &["d"]),
            ("Add", &["a", "d"], &["b"]),
            ("Add", &["b", "d"], &["c"]),
            ("Add", &["c", "a"], &["e"]),
            ("Add", &["e", "d"], &["y"]),
        ]);
        let mut profile = parse_test_profile(6_144, true, 10_000, 1, 1);
        profile
            .allowed_ops
            .get_mut(&("ai.onnx".to_string(), "Add".to_string()))
            .unwrap()
            .cpu_cost = None;

        let first = optimize(&model, &profile).unwrap();
        let second = optimize(&model, &profile).unwrap();
        let kinds = first.events.iter().map(|event| event.kind.as_str()).collect::<Vec<_>>();

        assert_eq!(first, second);
        assert_eq!(first.status, "pass");
        assert!(first.spill_bytes > 0);
        assert!(kinds.contains(&"spill"));
        assert!(kinds.contains(&"reload"));
        assert!(first.blockers.is_empty());
        assert!(first.peak_scratchpad_bytes <= 6_144);
    }

    #[test]
    fn fails_when_spill_is_disabled() {
        let model = parse_model(&[
            ("Relu", &["x"], &["a"]),
            ("Relu", &["bias"], &["d"]),
            ("Add", &["a", "d"], &["b"]),
            ("Add", &["b", "d"], &["c"]),
            ("Add", &["c", "a"], &["y"]),
        ]);
        let mut profile = parse_test_profile(3_072, false, 10_000, 1, 1);
        profile
            .allowed_ops
            .get_mut(&("ai.onnx".to_string(), "Add".to_string()))
            .unwrap()
            .cpu_cost = None;

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "fail");
        assert!(plan
            .blockers
            .iter()
            .any(|blocker| blocker.contains("scratchpad_unavailable")));
    }

    #[test]
    fn applies_trusted_replacement_recipe() {
        let model = parse_model(&[("HardSwish", &["x"], &["y"])]);
        let profile = parse_test_profile(262_144, true, 10_000, 10_000, 1);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "pass");
        assert_eq!(plan.assignments[0].device, "npu");
        assert_eq!(
            plan.assignments[0].recipe_id.as_deref(),
            Some("recipe.hardswish.v1")
        );
    }

    #[test]
    fn keeps_small_work_on_cpu_when_dma_cost_dominates() {
        let model = parse_model(&[("Relu", &["x"], &["y"])]);
        let profile = parse_test_profile(262_144, true, 1, 1, 100_000);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "pass");
        assert_eq!(plan.assignments[0].device, "cpu");
        assert!(plan.segments.is_empty());
        assert!(plan.events.is_empty());
    }

    fn parse_model(nodes: &[(&str, &[&str], &[&str])]) -> NormalizedModel {
        let mut tensor_names = BTreeSet::new();
        tensor_names.insert("x".to_string());
        tensor_names.insert("bias".to_string());
        for (_, inputs, outputs) in nodes {
            tensor_names.extend(inputs.iter().map(|name| (*name).to_string()));
            tensor_names.extend(outputs.iter().map(|name| (*name).to_string()));
        }
        let output = nodes
            .last()
            .and_then(|(_, _, outputs)| outputs.first())
            .copied()
            .unwrap();
        let values = tensor_names
            .iter()
            .filter(|name| {
                name.as_str() != "x" && name.as_str() != "bias" && name.as_str() != output
            })
            .map(|name| format!(r#"{{"name":"{name}","dtype":"int8","shape":[1,2048]}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let nodes = nodes
            .iter()
            .enumerate()
            .map(|(index, (op_type, inputs, outputs))| {
                let inputs = inputs
                    .iter()
                    .map(|name| format!(r#""{name}""#))
                    .collect::<Vec<_>>()
                    .join(",");
                let outputs = outputs
                    .iter()
                    .map(|name| format!(r#""{name}""#))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#"{{"name":"node_{index}","domain":"ai.onnx","op_type":"{op_type}","inputs":[{inputs}],"outputs":[{outputs}]}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        parse_normalized_model(&format!(
            r#"{{"schema":"edgefit.normalized_model.v1","model":{{"path":"tests/optimizer.onnx","file_bytes":1,"sha256":"sha256:optimizer-test"}},"graph":{{"inputs":[{{"name":"x","dtype":"int8","shape":[1,2048]}},{{"name":"bias","dtype":"int8","shape":[1,2048]}}],"values":[{values}],"outputs":[{{"name":"{output}","dtype":"int8","shape":[1,2048]}}],"initializers":[],"nodes":[{nodes}]}}}}"#
        ))
        .unwrap()
    }

    fn parse_test_profile(
        scratchpad_bytes: u64,
        spill_allowed: bool,
        relu_cpu_fixed_ns: u64,
        add_cpu_fixed_ns: u64,
        dma_setup_ns: u64,
    ) -> TargetProfile {
        parse_profile(
            &format!(
                r#"profile_version: edgefit.target.v1

metadata:
  source: EdgeFit optimizer unit-test simulation
  confidence: seed
  last_verified: 2026-07-12

target:
  id: optimizer_test
  name: Optimizer Test
  class: virtual-npu

memory:
  flash_bytes: 1048576
  ram_bytes: 1048576
  model_file_budget_bytes: 1048576
  peak_activation_budget_bytes: 1048576
  weights_residency: ram
  tensor_alignment_bytes: 1

runtime:
  name: optimizer-test
  static_shapes_required: true
  dynamic_allocation_allowed: false
  external_memory_allowed: true

dtype:
  allowed: [int8]
  preferred: int8
  fp32_allowed: false

opsets:
  ai.onnx: 18

accelerator:
  id: optimizer-test-npu
  confidence: seed-simulated
  scratchpad_bytes: {scratchpad_bytes}
  tensor_alignment_bytes: 1
  dma_burst_bytes: 1
  dma_setup_ns: {dma_setup_ns}
  dma_read_bytes_per_second: 1000000000
  dma_write_bytes_per_second: 1000000000
  spill_allowed: {spill_allowed}

ops:
  allow:
    ai.onnx:
      Relu:
        dtypes: [int8]
        cpu_cost:
          id: cpu.relu.int8
          kind: element
          fixed_ns: {relu_cpu_fixed_ns}
          throughput_per_second: 1000000000
        npu_cost:
          id: npu.relu.int8
          kind: element
          fixed_ns: 1
          throughput_per_second: 1000000000
      Add:
        dtypes: [int8]
        cpu_cost:
          id: cpu.add.int8
          kind: element
          fixed_ns: {add_cpu_fixed_ns}
          throughput_per_second: 1000000000
        npu_cost:
          id: npu.add.int8
          kind: element
          fixed_ns: 1
          throughput_per_second: 1000000000
      HardSigmoid:
        dtypes: [int8]
        npu_cost:
          id: npu.hardsigmoid.int8
          kind: element
          fixed_ns: 1
          throughput_per_second: 1000000000
      Mul:
        dtypes: [int8]
        npu_cost:
          id: npu.mul.int8
          kind: element
          fixed_ns: 1
          throughput_per_second: 1000000000
      HardSwish:
        dtypes: [int8]
        cpu_cost:
          id: cpu.hardswish.int8
          kind: element
          fixed_ns: 10000
          throughput_per_second: 1000000000

recipes:
  ai.onnx:
    HardSwish:
      id: recipe.hardswish.v1
      trusted: true
      source: EdgeFit optimizer unit-test simulation
      version: 1
      replacement_ops: [HardSigmoid, Mul]

shape:
  max_rank: 6
  allow_unknown_dims: false

quantization:
  required: true
  require_int8: true
  min_quantized_weight_fraction: 1.0
  min_operator_coverage: 1.0
"#
            ),
            PathBuf::from("tests/optimizer-profile.yaml"),
        )
        .unwrap()
    }

    #[test]
    fn fails_closed_when_target_has_no_accelerator() {
        let model = parse_normalized_model(include_str!(
            "../../../examples/models/virtual_npu_tiny.edgefit.json"
        ))
        .unwrap();
        let profile = parse_profile(
            include_str!("../../../targets/tflm-micro.yaml"),
            PathBuf::from("targets/tflm-micro.yaml"),
        )
        .unwrap();

        assert!(optimize(&model, &profile)
            .unwrap_err()
            .contains("does not declare an accelerator"));
    }
}
