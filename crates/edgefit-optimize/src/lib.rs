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

#[derive(Clone, Copy)]
struct Candidate<'a> {
    kernel_id: Option<&'a str>,
    launch_ns: u64,
    compute_ns: u64,
    recipe_id: Option<&'a str>,
}

pub fn optimize(model: &NormalizedModel, profile: &TargetProfile) -> EdgeFitResult<OptimizationPlan> {
    profile.validate()?;
    profile
        .accelerator
        .as_ref()
        .ok_or("target profile does not declare an accelerator")?;
    let producers = producers(model);
    let graph_outputs = model.outputs.iter().cloned().collect::<BTreeSet<_>>();
    let mut assignments = Vec::with_capacity(model.nodes.len());
    let mut cpu_assignments = Vec::with_capacity(model.nodes.len());
    let mut baseline_blockers = 0_u64;
    let mut baseline_latency = Some(0_u64);

    for (node_index, node) in model.nodes.iter().enumerate() {
        let rule = profile
            .op_rule(&node.domain, &node.op_type)
            .filter(|rule| contract_compatible(node, rule, model, profile));
        let cpu = match rule.and_then(|rule| rule.cpu_cost.as_ref()) {
            Some(cost) => evaluate_cost(cost, node, model, profile)?.map(|(launch_ns, compute_ns)| {
                Candidate {
                    kernel_id: Some(cost.id.as_str()),
                    launch_ns,
                    compute_ns,
                    recipe_id: None,
                }
            }),
            None => None,
        };
        let cpu_total = match cpu {
            Some(candidate) => Some(checked_add(
                candidate.launch_ns,
                candidate.compute_ns,
                "CPU node latency",
            )?),
            None => None,
        };
        baseline_latency = add_optional_latency(baseline_latency, cpu_total, "CPU baseline latency")?;
        if cpu.is_none() {
            baseline_blockers = checked_add(baseline_blockers, 1, "CPU baseline blocker count")?;
        }
        cpu_assignments.push(assignment_for_candidate(node_index, node, "cpu", cpu));

        let direct = match rule.and_then(|rule| rule.npu_cost.as_ref()) {
            Some(cost) => evaluate_cost(cost, node, model, profile)?.map(|(launch_ns, compute_ns)| {
                Candidate {
                    kernel_id: Some(cost.id.as_str()),
                    launch_ns,
                    compute_ns,
                    recipe_id: None,
                }
            }),
            None => None,
        };
        let recipe = match profile
            .replacement_recipes
            .get(&(node.domain.clone(), node.op_type.clone()))
        {
            Some(recipe) => recipe_candidate(recipe, node, model, profile)?,
            None => None,
        };
        let npu = best_candidate(direct, recipe)?;

        let choose_npu = match (cpu, npu) {
            (None, Some(_)) => true,
            (Some(cpu), Some(npu)) => {
                let boundary_ns = prospective_boundary_latency(
                    node,
                    model,
                    profile,
                    &producers,
                    &assignments,
                    &graph_outputs,
                )?;
                if let Some(boundary_ns) = boundary_ns {
                    let npu_total = checked_add(
                        checked_add(npu.launch_ns, npu.compute_ns, "NPU node latency")?,
                        boundary_ns,
                        "NPU node latency with boundary transfer",
                    )?;
                    let cpu_total = checked_add(cpu.launch_ns, cpu.compute_ns, "CPU node latency")?;
                    npu_total < cpu_total
                } else {
                    false
                }
            }
            _ => false,
        };

        if choose_npu {
            assignments.push(assignment_for_candidate(node_index, node, "npu", npu));
        } else {
            assignments.push(assignment_for_candidate(node_index, node, "cpu", cpu));
        }
    }

    let cpu_plan = build_plan(
        model,
        profile,
        baseline_blockers,
        baseline_latency,
        cpu_assignments,
    )?;
    let proposed_plan = build_plan(
        model,
        profile,
        baseline_blockers,
        baseline_latency,
        assignments,
    )?;
    let plan = if plan_is_better(&proposed_plan, &cpu_plan) {
        proposed_plan
    } else {
        cpu_plan
    };
    validate_plan_invariants(&plan, model, profile)?;
    Ok(plan)
}

fn prospective_boundary_latency(
    node: &NodeInfo,
    model: &NormalizedModel,
    profile: &TargetProfile,
    producers: &BTreeMap<String, usize>,
    assignments: &[NodeAssignment],
    graph_outputs: &BTreeSet<String>,
) -> EdgeFitResult<Option<u64>> {
    let accelerator = profile.accelerator.as_ref().ok_or("missing accelerator")?;
    let mut total = 0_u64;
    for name in node.inputs.iter().filter(|name| !name.is_empty()).filter(|name| {
        producers
            .get(*name)
            .and_then(|producer| assignments.get(*producer))
            .is_none_or(|assignment| assignment.device != "npu")
    }) {
        let Some(bytes) = tensor_bytes(model, profile, name)? else {
            return Ok(None);
        };
        let bytes = align_up(bytes, accelerator.dma_burst_bytes)?;
        if bytes != 0 {
            total = checked_add(
                total,
                dma_ns(
                    bytes,
                    accelerator.dma_setup_ns,
                    accelerator.dma_read_bytes_per_second,
                )?,
                "NPU boundary load latency",
            )?;
        }
    }
    for name in node.outputs.iter().filter(|name| graph_outputs.contains(*name)) {
        let Some(bytes) = tensor_bytes(model, profile, name)? else {
            return Ok(None);
        };
        let bytes = align_up(bytes, accelerator.dma_burst_bytes)?;
        if bytes != 0 {
            total = checked_add(
                total,
                dma_ns(
                    bytes,
                    accelerator.dma_setup_ns,
                    accelerator.dma_write_bytes_per_second,
                )?,
                "NPU boundary store latency",
            )?;
        }
    }
    Ok(Some(total))
}

fn best_candidate<'a>(
    direct: Option<Candidate<'a>>,
    recipe: Option<Candidate<'a>>,
) -> EdgeFitResult<Option<Candidate<'a>>> {
    match (direct, recipe) {
        (Some(direct), Some(recipe)) => {
            let direct_total = checked_add(
                direct.launch_ns,
                direct.compute_ns,
                "direct NPU candidate latency",
            )?;
            let recipe_total = checked_add(
                recipe.launch_ns,
                recipe.compute_ns,
                "replacement recipe candidate latency",
            )?;
            if recipe_total < direct_total {
                Ok(Some(recipe))
            } else {
                Ok(Some(direct))
            }
        }
        (candidate @ Some(_), None) | (None, candidate @ Some(_)) => Ok(candidate),
        (None, None) => Ok(None),
    }
}

fn plan_is_better(candidate: &OptimizationPlan, fallback: &OptimizationPlan) -> bool {
    if candidate.proposed_blockers != fallback.proposed_blockers {
        return candidate.proposed_blockers < fallback.proposed_blockers;
    }
    match (candidate.proposed_latency_ns, fallback.proposed_latency_ns) {
        (Some(candidate), Some(fallback)) => candidate < fallback,
        (Some(_), None) => true,
        _ => false,
    }
}

fn assignment_for_candidate(
    node_index: usize,
    node: &NodeInfo,
    device: &str,
    candidate: Option<Candidate<'_>>,
) -> NodeAssignment {
    match candidate {
        Some(candidate) => NodeAssignment {
            node_index,
            op_type: node.op_type.clone(),
            device: device.to_string(),
            kernel_id: candidate.kernel_id.map(str::to_string),
            recipe_id: candidate.recipe_id.map(str::to_string),
            launch_ns: Some(candidate.launch_ns),
            compute_ns: Some(candidate.compute_ns),
        },
        None => NodeAssignment {
            node_index,
            op_type: node.op_type.clone(),
            device: "unsupported".to_string(),
            kernel_id: None,
            recipe_id: None,
            launch_ns: None,
            compute_ns: None,
        },
    }
}

fn build_plan(
    model: &NormalizedModel,
    profile: &TargetProfile,
    baseline_blockers: u64,
    baseline_latency_ns: Option<u64>,
    assignments: Vec<NodeAssignment>,
) -> EdgeFitResult<OptimizationPlan> {
    let accelerator = profile.accelerator.as_ref().ok_or("missing accelerator")?;
    let segments = collect_segments(&assignments);
    let simulation = simulate_npu(model, profile, &assignments, &segments)?;
    let launch_latency_ns = sum_optional(
        assignments.iter().map(|assignment| assignment.launch_ns),
        "proposed launch latency",
    )?;
    let compute_latency_ns = sum_optional(
        assignments.iter().map(|assignment| assignment.compute_ns),
        "proposed compute latency",
    )?;
    let proposed_latency_ns = match launch_latency_ns.zip(compute_latency_ns) {
        Some((launch, compute)) => Some(checked_add(
            checked_add(launch, compute, "proposed kernel latency")?,
            simulation.transfer_ns,
            "proposed total latency",
        )?),
        None => None,
    };
    let mut blockers = assignments
        .iter()
        .filter(|assignment| assignment.device == "unsupported")
        .map(|assignment| format!("node:{} unsupported", assignment.node_index))
        .collect::<Vec<_>>();
    blockers.extend(simulation.blockers);
    if proposed_latency_ns.is_none() {
        blockers.push("latency_unknown".to_string());
    }
    let proposed_blockers = u64::try_from(blockers.len())
        .map_err(|_| "arithmetic overflow converting proposed blocker count".to_string())?;
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
        baseline_latency_ns,
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
    assignments: &[NodeAssignment],
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
            let mut protected = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .cloned()
                .collect::<BTreeSet<_>>();
            for input in node.inputs.iter().filter(|name| !name.is_empty()) {
                if resident.contains_key(input) {
                    continue;
                }
                let Some(bytes) = tensor_bytes(model, profile, input)? else {
                    blockers.push(format!("node:{node_index} tensor:{input} size_unknown"));
                    continue;
                };
                let bytes = align_up(bytes, accelerator.tensor_alignment_bytes)?;
                if spill_until_fit(
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
                )? {
                    if resident.insert(input.clone(), bytes).is_some() {
                        return Err(format!(
                            "scratchpad accounting encountered duplicate resident tensor {input}"
                        ));
                    }
                    used = checked_add(used, bytes, "scratchpad input allocation")?;
                    peak_bytes = peak_bytes.max(used);
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
                    )?;
                } else {
                    blockers.push(scratchpad_blocker(
                        node_index,
                        Some(input),
                        accelerator.spill_allowed,
                    ));
                }
            }

            let workspace_bytes = assignment_workspace_bytes(node, &assignments[node_index], profile)?;
            let workspace_bytes = align_up(workspace_bytes, accelerator.tensor_alignment_bytes)?;
            let workspace_allocated = if workspace_bytes == 0 {
                true
            } else if spill_until_fit(
                node_index,
                workspace_bytes,
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
            )? {
                used = checked_add(used, workspace_bytes, "NPU workspace allocation")?;
                peak_bytes = peak_bytes.max(used);
                true
            } else {
                blockers.push(scratchpad_blocker(
                    node_index,
                    None,
                    accelerator.spill_allowed,
                ));
                false
            };

            for output in node.outputs.iter().filter(|name| !name.is_empty()) {
                let Some(bytes) = tensor_bytes(model, profile, output)? else {
                    blockers.push(format!("node:{node_index} tensor:{output} size_unknown"));
                    continue;
                };
                let bytes = align_up(bytes, accelerator.tensor_alignment_bytes)?;
                if spill_until_fit(
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
                )? {
                    if resident.insert(output.clone(), bytes).is_some() {
                        return Err(format!(
                            "scratchpad accounting encountered duplicate resident tensor {output}"
                        ));
                    }
                    used = checked_add(used, bytes, "scratchpad output allocation")?;
                    peak_bytes = peak_bytes.max(used);
                    protected.insert(output.clone());
                } else {
                    blockers.push(scratchpad_blocker(
                        node_index,
                        Some(output),
                        accelerator.spill_allowed,
                    ));
                }
            }
            if workspace_allocated {
                used = checked_sub(used, workspace_bytes, "NPU workspace release")?;
            }

            let dead = resident
                .iter()
                .filter(|(tensor, _)| {
                    !consumers
                        .get(*tensor)
                        .is_some_and(|uses| {
                            uses.iter().any(|index| {
                                *index > node_index && *index <= segment.last_node
                            })
                        })
                        && !graph_outputs.contains(*tensor)
                        && !consumers
                            .get(*tensor)
                            .is_some_and(|uses| uses.iter().any(|index| *index > segment.last_node))
                })
                .map(|(tensor, bytes)| (tensor.clone(), *bytes))
                .collect::<Vec<_>>();
            let mut released = 0_u64;
            for (tensor, bytes) in dead {
                resident.remove(&tensor);
                released = checked_add(released, bytes, "scratchpad dead tensor byte sum")?;
            }
            used = checked_sub(used, released, "scratchpad dead tensor release")?;
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
            )?;
        }
    }

    Ok(Simulation { events, transfer_ns, transfer_bytes, spill_bytes, peak_bytes, blockers })
}

fn assignment_workspace_bytes(
    node: &NodeInfo,
    assignment: &NodeAssignment,
    profile: &TargetProfile,
) -> EdgeFitResult<u64> {
    if assignment.device != "npu" {
        return Ok(0);
    }
    let Some(recipe_id) = assignment.recipe_id.as_deref() else {
        return Ok(profile
            .op_rule(&node.domain, &node.op_type)
            .map_or(0, |rule| rule.workspace_bytes));
    };
    let recipe = profile
        .replacement_recipes
        .get(&(node.domain.clone(), node.op_type.clone()))
        .filter(|recipe| recipe.id == recipe_id)
        .ok_or_else(|| format!("missing replacement recipe {recipe_id} for workspace accounting"))?;
    let mut workspace_bytes = 0_u64;
    for replacement in &recipe.replacement_ops {
        let rule = profile
            .op_rule(&node.domain, replacement)
            .ok_or_else(|| format!("missing replacement operator {replacement} for workspace accounting"))?;
        workspace_bytes = workspace_bytes.max(rule.workspace_bytes);
    }
    Ok(workspace_bytes)
}

fn scratchpad_blocker(node_index: usize, tensor: Option<&str>, spill_allowed: bool) -> String {
    let subject = tensor
        .map(|tensor| format!("tensor:{tensor}"))
        .unwrap_or_else(|| "workspace".to_string());
    if spill_allowed {
        format!("node:{node_index} {subject} scratchpad_unavailable")
    } else {
        format!("node:{node_index} {subject} scratchpad_unavailable spill_disabled")
    }
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
) -> EdgeFitResult<bool> {
    if checked_add(*used, required, "scratchpad capacity check")?
        <= accelerator.scratchpad_bytes
    {
        return Ok(true);
    }
    if !accelerator.spill_allowed {
        return Ok(false);
    }

    let mut planned_used = *used;
    let mut victims = Vec::<(String, u64)>::new();
    let mut selected = BTreeSet::<String>::new();
    while checked_add(planned_used, required, "scratchpad capacity check")?
        > accelerator.scratchpad_bytes
    {
        let victim = resident
            .iter()
            .filter(|(tensor, _)| !protected.contains(*tensor) && !selected.contains(*tensor))
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
            return Ok(false);
        };
        planned_used = checked_sub(planned_used, bytes, "planned scratchpad spill release")?;
        selected.insert(tensor.clone());
        victims.push((tensor, bytes));
    }

    for (tensor, bytes) in victims {
        resident.remove(&tensor);
        spilled.insert(tensor.clone());
        *used = checked_sub(*used, bytes, "scratchpad spill release")?;
        let transferred_bytes = align_up(bytes, accelerator.dma_burst_bytes)?;
        *spill_bytes = checked_add(*spill_bytes, transferred_bytes, "spill byte total")?;
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
        )?;
    }
    Ok(true)
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
) -> EdgeFitResult<()> {
    let bytes = align_up(bytes, accelerator.dma_burst_bytes)?;
    if bytes == 0 {
        return Ok(());
    }
    let latency_ns = dma_ns(bytes, accelerator.dma_setup_ns, bandwidth)?;
    *transfer_ns = checked_add(*transfer_ns, latency_ns, "transfer latency total")?;
    *transfer_bytes = checked_add(*transfer_bytes, bytes, "transfer byte total")?;
    events.push(TransferEvent {
        kind: kind.to_string(),
        tensor: tensor.to_string(),
        bytes,
        at_node,
        latency_ns,
    });
    Ok(())
}

fn recipe_candidate<'a>(
    recipe: &'a ReplacementRecipe,
    node: &NodeInfo,
    model: &NormalizedModel,
    profile: &'a TargetProfile,
) -> EdgeFitResult<Option<Candidate<'a>>> {
    if !recipe.trusted
        || profile
            .op_rule(&node.domain, &node.op_type)
            .is_some_and(|rule| !contract_compatible(node, rule, model, profile))
    {
        return Ok(None);
    }
    if recipe.id.trim().is_empty()
        || recipe.version.trim().is_empty()
        || recipe.source.trim().is_empty()
        || recipe.replacement_ops.is_empty()
    {
        return Ok(None);
    }
    let mut launch_ns = 0_u64;
    let mut compute_ns = 0_u64;
    for replacement in &recipe.replacement_ops {
        let Some(rule) = profile.op_rule(&node.domain, replacement) else {
            return Ok(None);
        };
        if !contract_compatible(node, rule, model, profile) {
            return Ok(None);
        }
        let Some(cost) = rule.npu_cost.as_ref() else {
            return Ok(None);
        };
        let Some((launch, compute)) = evaluate_cost(cost, node, model, profile)? else {
            return Ok(None);
        };
        launch_ns = checked_add(launch_ns, launch, "replacement recipe launch latency")?;
        compute_ns = checked_add(compute_ns, compute, "replacement recipe compute latency")?;
    }
    Ok(Some(Candidate {
        kernel_id: None,
        launch_ns,
        compute_ns,
        recipe_id: Some(recipe.id.as_str()),
    }))
}

fn evaluate_cost(
    cost: &KernelCost,
    node: &NodeInfo,
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> EdgeFitResult<Option<(u64, u64)>> {
    let units = match cost.kind.as_str() {
        "fixed" => Some(0),
        "element" => match node.outputs.first() {
            Some(name) => tensor_elements(model, profile, name)?,
            None => None,
        },
        "bytes" => {
            let mut total = 0_u64;
            let mut known = true;
            for name in node.inputs.iter().chain(node.outputs.iter()).filter(|name| !name.is_empty()) {
                let Some(bytes) = tensor_bytes(model, profile, name)? else {
                    known = false;
                    break;
                };
                total = checked_add(total, bytes, "kernel byte cost units")?;
            }
            known.then_some(total)
        }
        "mac" => mac_units(node, model, profile)?,
        _ => None,
    };
    let Some(units) = units else {
        return Ok(None);
    };
    let compute_ns = if units == 0 {
        0
    } else {
        ceil_mul_div(units, 1_000_000_000, cost.throughput_per_second)?
    };
    Ok(Some((cost.fixed_ns, compute_ns)))
}

fn mac_units(
    node: &NodeInfo,
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> EdgeFitResult<Option<u64>> {
    match node.op_type.as_str() {
        "MatMul" | "Gemm" => {
            let Some(left) = node
                .inputs
                .first()
                .and_then(|name| model.tensors.get(name))
                .and_then(|tensor| tensor.shape.as_ref())
            else {
                return Ok(None);
            };
            let Some(right) = node
                .inputs
                .get(1)
                .and_then(|name| model.tensors.get(name))
                .and_then(|tensor| tensor.shape.as_ref())
            else {
                return Ok(None);
            };
            let Some(m) = left
                .len()
                .checked_sub(2)
                .and_then(|index| left.get(index))
                .and_then(|dim| bounded_dim(dim, profile))
            else {
                return Ok(None);
            };
            let Some(k) = left.last().and_then(|dim| bounded_dim(dim, profile)) else {
                return Ok(None);
            };
            let Some(n) = right.last().and_then(|dim| bounded_dim(dim, profile)) else {
                return Ok(None);
            };
            Ok(Some(checked_mul(
                checked_mul(m, k, "matrix MAC units")?,
                n,
                "matrix MAC units",
            )?))
        }
        "Conv" => {
            let Some(output) = node.outputs.first() else {
                return Ok(None);
            };
            let Some(output_elements) = tensor_elements(model, profile, output)? else {
                return Ok(None);
            };
            let Some(input) = node
                .inputs
                .first()
                .and_then(|name| model.tensors.get(name))
                .and_then(|tensor| tensor.shape.as_ref())
            else {
                return Ok(None);
            };
            let Some(channels) = input.get(1).and_then(|dim| bounded_dim(dim, profile)) else {
                return Ok(None);
            };
            let group = match node.attributes.get("group") {
                Some(AttributeValue::Int(value)) if *value > 0 => u64::try_from(*value)
                    .map_err(|_| "arithmetic overflow converting convolution group".to_string())?,
                _ => 1,
            };
            let kernel = match node.attributes.get("kernel_shape") {
                Some(AttributeValue::Ints(values)) => {
                    let mut total = 1_u64;
                    for value in values {
                        if *value <= 0 {
                            return Ok(None);
                        }
                        let value = u64::try_from(*value).map_err(|_| {
                            "arithmetic overflow converting convolution kernel dimension".to_string()
                        })?;
                        total = checked_mul(total, value, "convolution kernel elements")?;
                    }
                    total
                }
                _ => 1,
            };
            if group > channels || channels % group != 0 {
                return Ok(None);
            }
            Ok(Some(checked_mul(
                checked_mul(output_elements, channels / group, "convolution MAC units")?,
                kernel,
                "convolution MAC units",
            )?))
        }
        _ => Ok(None),
    }
}

fn contract_compatible(
    node: &NodeInfo,
    rule: &OpRule,
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> bool {
    for (ports, names) in [(&rule.input_dtypes, &node.inputs), (&rule.output_dtypes, &node.outputs)] {
        if ports.keys().any(|port| names.get(*port).is_none_or(String::is_empty)) {
            return false;
        }
        for (port, name) in names.iter().enumerate().filter(|(_, name)| !name.is_empty()) {
            let Some(dtype) = model.tensors.get(name).and_then(|tensor| tensor.dtype.as_ref()) else {
                return false;
            };
            let allowed = ports.get(&port).unwrap_or(&rule.dtypes);
            if !allowed.contains(dtype) {
                return false;
            }
        }
    }
    let effective_max_rank = match (rule.max_rank, profile.shape_max_rank) {
        (Some(rule), Some(global)) => Some(rule.min(global)),
        (Some(rule), None) => Some(rule),
        (None, Some(global)) => Some(global),
        (None, None) => None,
    };
    if let Some(max_rank) = effective_max_rank {
        for name in node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .filter(|name| !name.is_empty())
        {
            let Some(rank) = model
                .tensors
                .get(name)
                .and_then(|tensor| tensor.shape.as_ref())
                .and_then(|shape| u64::try_from(shape.len()).ok())
            else {
                return false;
            };
            if rank > max_rank {
                return false;
            }
        }
    }
    for (name, allowed) in &rule.attributes {
        if !node.attributes.get(name).is_some_and(|value| allowed.contains(value)) {
            return false;
        }
    }
    true
}

fn validate_plan_invariants(
    plan: &OptimizationPlan,
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> EdgeFitResult<()> {
    let invalid = |message: &str| Err(format!("invalid optimization plan invariant: {message}"));
    if plan.schema != "edgefit.optimization_plan.v1" {
        return invalid("schema changed");
    }
    let accelerator = profile.accelerator.as_ref().ok_or("missing accelerator")?;
    if plan.model_sha256 != model.sha256
        || plan.target_id != profile.target_id
        || plan.target_fingerprint != profile.fingerprint
        || plan.accelerator_id != accelerator.id
        || plan.confidence != accelerator.confidence
    {
        return invalid("model or target identity is inconsistent");
    }
    if plan.assignments.len() != model.nodes.len()
        || plan
            .assignments
            .iter()
            .enumerate()
            .any(|(index, assignment)| {
                assignment.node_index != index || assignment.op_type != model.nodes[index].op_type
            })
    {
        return invalid("assignments are not contiguous model-node assignments");
    }
    if plan.segments != collect_segments(&plan.assignments) {
        return invalid("segments are not exactly the contiguous NPU runs");
    }
    if plan.peak_scratchpad_bytes > accelerator.scratchpad_bytes {
        return invalid("peak scratchpad exceeds accelerator capacity");
    }

    let mut transfer_ns = 0_u64;
    let mut transfer_bytes = 0_u64;
    let mut spill_bytes = 0_u64;
    let mut active_segment = None;
    let mut spilled = BTreeSet::<String>::new();
    for event in &plan.events {
        if !matches!(event.kind.as_str(), "load" | "store" | "spill" | "reload") {
            return invalid("event has an unknown kind");
        }
        if event.at_node >= model.nodes.len()
            || plan.assignments[event.at_node].device != "npu"
            || !model.tensors.contains_key(&event.tensor)
            || event.bytes % accelerator.dma_burst_bytes != 0
        {
            return invalid("event has an invalid node, tensor, or byte reference");
        }
        let segment_id = plan
            .segments
            .iter()
            .position(|segment| {
                event.at_node >= segment.first_node && event.at_node <= segment.last_node
            })
            .ok_or_else(|| {
                "invalid optimization plan invariant: event is outside every NPU segment".to_string()
            })?;
        if let Some(previous_segment) = active_segment {
            if segment_id < previous_segment {
                return invalid("events are not ordered by NPU segment");
            }
        }
        if active_segment != Some(segment_id) {
            active_segment = Some(segment_id);
            spilled.clear();
        }
        let bandwidth = if matches!(event.kind.as_str(), "store" | "spill") {
            accelerator.dma_write_bytes_per_second
        } else {
            accelerator.dma_read_bytes_per_second
        };
        if event.latency_ns != dma_ns(event.bytes, accelerator.dma_setup_ns, bandwidth)? {
            return invalid("event latency is inconsistent with its transfer");
        }
        match event.kind.as_str() {
            "spill" => {
                if !accelerator.spill_allowed || !spilled.insert(event.tensor.clone()) {
                    return invalid("spill event is illegal");
                }
                spill_bytes = checked_add(spill_bytes, event.bytes, "invariant spill byte total")?;
            }
            "reload" => {
                if !spilled.remove(&event.tensor) {
                    return invalid("reload does not reference a preceding unmatched spill");
                }
            }
            _ => {}
        }
        transfer_ns = checked_add(transfer_ns, event.latency_ns, "invariant transfer latency")?;
        transfer_bytes = checked_add(transfer_bytes, event.bytes, "invariant transfer bytes")?;
    }
    if !accelerator.spill_allowed && plan.spill_bytes != 0 {
        return invalid("spill bytes are non-zero when spilling is disabled");
    }
    if transfer_ns != plan.transfer_latency_ns
        || transfer_bytes != plan.transfer_bytes
        || spill_bytes != plan.spill_bytes
    {
        return invalid("event totals do not match plan totals");
    }

    let launch = sum_optional(
        plan.assignments.iter().map(|assignment| assignment.launch_ns),
        "invariant launch latency",
    )?;
    let compute = sum_optional(
        plan.assignments.iter().map(|assignment| assignment.compute_ns),
        "invariant compute latency",
    )?;
    if launch != plan.launch_latency_ns || compute != plan.compute_latency_ns {
        return invalid("assignment latency totals do not match plan totals");
    }
    let proposed = match launch.zip(compute) {
        Some((launch, compute)) => Some(checked_add(
            checked_add(launch, compute, "invariant kernel latency")?,
            transfer_ns,
            "invariant proposed latency",
        )?),
        None => None,
    };
    if proposed != plan.proposed_latency_ns {
        return invalid("proposed latency is inconsistent");
    }
    let mut baseline_blockers = 0_u64;
    let mut baseline_latency = Some(0_u64);
    for node in &model.nodes {
        let cpu = profile
            .op_rule(&node.domain, &node.op_type)
            .filter(|rule| contract_compatible(node, rule, model, profile))
            .and_then(|rule| rule.cpu_cost.as_ref());
        let cpu_latency = match cpu {
            Some(cost) => evaluate_cost(cost, node, model, profile)?.map(|(launch, compute)| {
                checked_add(launch, compute, "invariant CPU node latency")
            }).transpose()?,
            None => None,
        };
        if cpu_latency.is_none() {
            baseline_blockers = checked_add(
                baseline_blockers,
                1,
                "invariant CPU baseline blocker count",
            )?;
        }
        baseline_latency = add_optional_latency(
            baseline_latency,
            cpu_latency,
            "invariant CPU baseline latency",
        )?;
    }
    if plan.baseline_blockers != baseline_blockers || plan.baseline_latency_ns != baseline_latency {
        return invalid("CPU baseline is inconsistent");
    }

    let blocker_count = u64::try_from(plan.blockers.len())
        .map_err(|_| "arithmetic overflow converting invariant blocker count".to_string())?;
    if blocker_count != plan.proposed_blockers
        || (plan.status == "pass") != plan.blockers.is_empty()
        || !matches!(plan.status.as_str(), "pass" | "fail")
        || (plan.proposed_latency_ns.is_none()) != plan.blockers.iter().any(|item| item == "latency_unknown")
    {
        return invalid("status, blocker count, or latency state is inconsistent");
    }
    for assignment in &plan.assignments {
        let node = &model.nodes[assignment.node_index];
        match assignment.device.as_str() {
            "cpu"
                if assignment.kernel_id.is_some()
                    && assignment.recipe_id.is_none()
                    && assignment.launch_ns.is_some()
                    && assignment.compute_ns.is_some() => {}
            "npu"
                if (assignment.kernel_id.is_some() ^ assignment.recipe_id.is_some())
                    && assignment.launch_ns.is_some()
                    && assignment.compute_ns.is_some() => {}
            "unsupported"
                if assignment.kernel_id.is_none()
                    && assignment.recipe_id.is_none()
                    && assignment.launch_ns.is_none()
                    && assignment.compute_ns.is_none()
                    && plan
                        .blockers
                        .contains(&format!("node:{} unsupported", assignment.node_index)) => {}
            _ => return invalid("assignment support and latency state is inconsistent"),
        }
        if assignment.device != "npu" && assignment.recipe_id.is_some() {
            return invalid("recipe is attached to a non-NPU assignment");
        }
        if assignment.device == "cpu" {
            let candidate = match profile
                .op_rule(&node.domain, &node.op_type)
                .filter(|rule| contract_compatible(node, rule, model, profile))
                .and_then(|rule| rule.cpu_cost.as_ref())
            {
                Some(cost) => evaluate_cost(cost, node, model, profile)?.map(|(launch, compute)| {
                    (cost.id.as_str(), launch, compute)
                }),
                None => None,
            };
            if !candidate.is_some_and(|(kernel, launch, compute)| {
                assignment.kernel_id.as_deref() == Some(kernel)
                    && assignment.launch_ns == Some(launch)
                    && assignment.compute_ns == Some(compute)
            }) {
                return invalid("CPU assignment does not satisfy its kernel contract");
            }
        } else if assignment.device == "npu" {
            let direct = match profile
                .op_rule(&node.domain, &node.op_type)
                .filter(|rule| contract_compatible(node, rule, model, profile))
                .and_then(|rule| rule.npu_cost.as_ref())
            {
                Some(cost) => evaluate_cost(cost, node, model, profile)?.map(|(launch, compute)| {
                    (cost.id.as_str(), launch, compute)
                }),
                None => None,
            };
            let direct_valid = direct.is_some_and(|(kernel, launch, compute)| {
                assignment.kernel_id.as_deref() == Some(kernel)
                    && assignment.launch_ns == Some(launch)
                    && assignment.compute_ns == Some(compute)
            });
            let recipe = match assignment.recipe_id.as_deref().and_then(|recipe_id| {
                profile
                    .replacement_recipes
                    .get(&(node.domain.clone(), node.op_type.clone()))
                    .filter(|recipe| recipe.id == recipe_id)
            }) {
                Some(recipe) => recipe_candidate(recipe, node, model, profile)?,
                None => None,
            };
            let recipe_valid = recipe.is_some_and(|candidate| {
                assignment.kernel_id.is_none()
                    && assignment.recipe_id.as_deref() == candidate.recipe_id
                    && assignment.launch_ns == Some(candidate.launch_ns)
                    && assignment.compute_ns == Some(candidate.compute_ns)
            });
            if !direct_valid && !recipe_valid {
                return invalid("NPU assignment does not satisfy a kernel or recipe contract");
            }
        }
    }
    if plan.plan_hash
        != plan_fingerprint(
            &plan.model_sha256,
            &plan.target_fingerprint,
            &plan.assignments,
            &plan.segments,
            &plan.events,
        )
    {
        return invalid("plan hash is inconsistent");
    }
    Ok(())
}

fn collect_segments(assignments: &[NodeAssignment]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut start = None;
    for (index, assignment) in assignments.iter().enumerate() {
        if assignment.device == "npu" {
            start.get_or_insert(index);
        } else if let Some(first) = start.take() {
            let last_node = index
                .checked_sub(1)
                .expect("an open NPU segment always has a preceding assignment");
            segments.push(Segment { id: segments.len(), first_node: first, last_node });
        }
    }
    if let Some(first) = start {
        let last_node = assignments
            .len()
            .checked_sub(1)
            .expect("an open NPU segment requires a non-empty assignment list");
        segments.push(Segment { id: segments.len(), first_node: first, last_node });
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

fn tensor_bytes(
    model: &NormalizedModel,
    profile: &TargetProfile,
    name: &str,
) -> EdgeFitResult<Option<u64>> {
    let Some(tensor) = model.tensors.get(name) else {
        return Ok(None);
    };
    if let Some(bytes) = tensor.bytes {
        return Ok(Some(bytes));
    }
    let Some(dtype) = tensor.dtype.as_deref() else {
        return Ok(None);
    };
    let Some(dtype_bytes) = edgefit_ir::dtype_bytes(dtype) else {
        return Ok(None);
    };
    let Some(shape) = tensor.shape.as_ref() else {
        return Ok(None);
    };
    let mut bytes = dtype_bytes;
    for dim in shape {
        let Some(value) = bounded_dim(dim, profile) else {
            return Ok(None);
        };
        bytes = checked_mul(bytes, value, "tensor byte size")?;
    }
    Ok(Some(bytes))
}

fn tensor_elements(
    model: &NormalizedModel,
    profile: &TargetProfile,
    name: &str,
) -> EdgeFitResult<Option<u64>> {
    let Some(shape) = model.tensors.get(name).and_then(|tensor| tensor.shape.as_ref()) else {
        return Ok(None);
    };
    let mut total = 1_u64;
    for dim in shape {
        let Some(value) = bounded_dim(dim, profile) else {
            return Ok(None);
        };
        total = checked_mul(total, value, "tensor element count")?;
    }
    Ok(Some(total))
}

fn bounded_dim(dim: &Dim, profile: &TargetProfile) -> Option<u64> {
    match dim {
        Dim::Known(value) if *value >= 0 => u64::try_from(*value).ok(),
        Dim::Symbol(symbol) => profile.symbol_bounds.get(symbol).copied(),
        _ => None,
    }
}

fn add_optional_latency(
    current: Option<u64>,
    next: Option<u64>,
    context: &str,
) -> EdgeFitResult<Option<u64>> {
    match current.zip(next) {
        Some((left, right)) => Ok(Some(checked_add(left, right, context)?)),
        None => Ok(None),
    }
}

fn sum_optional(
    values: impl Iterator<Item = Option<u64>>,
    context: &str,
) -> EdgeFitResult<Option<u64>> {
    let mut total = Some(0_u64);
    for value in values {
        total = add_optional_latency(total, value, context)?;
    }
    Ok(total)
}

fn checked_add(left: u64, right: u64, context: &str) -> EdgeFitResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| format!("arithmetic overflow computing {context}"))
}

fn checked_sub(left: u64, right: u64, context: &str) -> EdgeFitResult<u64> {
    left.checked_sub(right)
        .ok_or_else(|| format!("arithmetic underflow computing {context}"))
}

fn checked_mul(left: u64, right: u64, context: &str) -> EdgeFitResult<u64> {
    left.checked_mul(right)
        .ok_or_else(|| format!("arithmetic overflow computing {context}"))
}

fn align_up(value: u64, alignment: u64) -> EdgeFitResult<u64> {
    if alignment == 0 {
        return Err("alignment must be greater than zero".to_string());
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        checked_add(value, alignment - remainder, "aligned byte size")
    }
}

fn dma_ns(bytes: u64, setup_ns: u64, bandwidth: u64) -> EdgeFitResult<u64> {
    checked_add(
        setup_ns,
        ceil_mul_div(bytes, 1_000_000_000, bandwidth)?,
        "DMA latency",
    )
}

fn ceil_mul_div(value: u64, multiplier: u64, divisor: u64) -> EdgeFitResult<u64> {
    if divisor == 0 {
        return Err("cannot compute latency with zero throughput".to_string());
    }
    let numerator = u128::from(value)
        .checked_mul(u128::from(multiplier))
        .ok_or_else(|| "arithmetic overflow computing scaled latency".to_string())?;
    let result = numerator
        .checked_add(u128::from(divisor - 1))
        .ok_or_else(|| "arithmetic overflow computing rounded latency".to_string())?
        / u128::from(divisor);
    u64::try_from(result).map_err(|_| "arithmetic overflow converting latency".to_string())
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

    #[test]
    fn contract_mismatch_does_not_bypass_cpu_fallback() {
        let mut model = parse_model(&[("Relu", &["x"], &["y"])]);
        model.tensors.get_mut("x").unwrap().dtype = Some("float32".to_string());
        model.tensors.get_mut("y").unwrap().dtype = Some("float32".to_string());
        let profile = parse_test_profile(262_144, true, 10_000, 10_000, 1);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "fail");
        assert_eq!(plan.assignments[0].device, "unsupported");
        assert_eq!(plan.baseline_blockers, 1);
        assert!(plan.blockers.contains(&"node:0 unsupported".to_string()));
    }

    #[test]
    fn port_dtype_override_and_rank_limit_fail_closed() {
        let model = parse_model(&[("Add", &["x", "bias"], &["y"])]);
        let mut profile = parse_test_profile(262_144, true, 10_000, 10_000, 1);
        let rule = profile
            .allowed_ops
            .get_mut(&("ai.onnx".to_string(), "Add".to_string()))
            .unwrap();
        rule.input_dtypes.insert(1, ["float32".to_string()].into());
        rule.max_rank = Some(1);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "fail");
        assert_eq!(plan.assignments[0].device, "unsupported");
    }

    #[test]
    fn workspace_pressure_is_a_canonical_capacity_failure() {
        let model = parse_model(&[("Relu", &["x"], &["y"])]);
        let mut profile = parse_test_profile(4_096, false, 10_000, 10_000, 1);
        profile
            .allowed_ops
            .get_mut(&("ai.onnx".to_string(), "Relu".to_string()))
            .unwrap()
            .workspace_bytes = 4_096;

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.status, "fail");
        assert!(plan
            .blockers
            .iter()
            .any(|blocker| blocker.contains("workspace scratchpad_unavailable")));
        assert!(plan.peak_scratchpad_bytes <= 4_096);
    }

    #[test]
    fn known_cpu_baseline_wins_when_mixed_plan_is_not_faster() {
        let model = parse_model(&[("Relu", &["x"], &["y"])]);
        let profile = parse_test_profile(262_144, true, 1, 1, 100_000);

        let plan = optimize(&model, &profile).unwrap();

        assert_eq!(plan.proposed_blockers, 0);
        assert_eq!(plan.proposed_latency_ns, plan.baseline_latency_ns);
        assert_eq!(plan.assignments[0].device, "cpu");
    }

    #[test]
    fn arithmetic_overflow_is_an_execution_error() {
        let model = parse_model(&[("Relu", &["x"], &["y"])]);
        let mut profile = parse_test_profile(262_144, true, 10_000, 10_000, 1);
        let cost = profile
            .allowed_ops
            .get_mut(&("ai.onnx".to_string(), "Relu".to_string()))
            .unwrap()
            .cpu_cost
            .as_mut()
            .unwrap();
        cost.fixed_ns = u64::MAX;

        let error = optimize(&model, &profile).unwrap_err();

        assert!(error.contains("arithmetic overflow"));
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
