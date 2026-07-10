//! Activation arena 的确定性规划器。
//!
//! 本模块用索引化生命周期和双索引 best-fit free list 计算逻辑峰值与物理
//! arena 高水位；只有 target profile 明确授权且满足安全条件时才执行原地复用。

use edgefit_ir::{NormalizedModel, TensorInfo};
use edgefit_target::TargetProfile;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Bound::{Excluded, Unbounded};

const TOP_CONTRIBUTOR_LIMIT: usize = 8;
const PLANNER_ALGORITHM: &str = "linear_scan_best_fit_v2";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeakMemoryContributor {
    pub tensor: String,
    pub allocated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAllocationTrace {
    pub kind: String,
    pub name: String,
    pub logical_bytes: Option<u64>,
    pub allocated_bytes: Option<u64>,
    pub arena_offset: Option<u64>,
    pub first_event: u64,
    pub last_event: u64,
    pub alias_of: Option<String>,
    pub size_source: String,
    pub graph_output: bool,
}

pub(crate) struct ActivationMemoryPlan {
    pub logical_peak_bytes: u64,
    pub planned_arena_bytes: u64,
    pub alignment_bytes: u64,
    pub algorithm: String,
    pub planning_overflowed: bool,
    pub peak_event: String,
    pub peak_node_index: Option<u64>,
    pub peak_node_name: Option<String>,
    pub peak_op_type: Option<String>,
    pub peak_live_allocated_bytes: u64,
    pub peak_workspace_bytes: u64,
    pub peak_fragmentation_bytes: u64,
    pub inplace_reuse_count: u64,
    pub inplace_avoided_allocation_bytes: u64,
    pub peak_contributors: Vec<PeakMemoryContributor>,
    pub allocation_trace: Vec<MemoryAllocationTrace>,
    pub confidence: String,
    pub bounded_dynamic_tensor_count: u64,
    pub unresolved_tensor_size_count: u64,
}

#[derive(Clone)]
struct TensorFact {
    logical_bytes: Option<u64>,
    allocated_bytes: Option<u64>,
    size_source: String,
    used_profile_bound: bool,
    alignment_overflowed: bool,
}

struct ActiveTensor {
    logical_bytes: u64,
    slot_id: Option<usize>,
    trace_index: usize,
}

struct SlotState {
    offset: u64,
    capacity: u64,
    reference_count: u64,
}

#[derive(Default)]
struct Arena {
    free_by_offset: BTreeMap<u64, u64>,
    free_by_size: BTreeSet<(u64, u64)>,
    capacity: u64,
}

impl Arena {
    fn allocate(&mut self, bytes: u64) -> Option<u64> {
        if bytes == 0 {
            return Some(0);
        }
        let candidate = self
            .free_by_size
            .range((bytes, 0)..)
            .next()
            .copied();
        if let Some((block_bytes, offset)) = candidate {
            self.remove_free_block(offset, block_bytes);
            if block_bytes > bytes {
                self.insert_free_block(offset + bytes, block_bytes - bytes);
            }
            return Some(offset);
        }

        let offset = self.capacity;
        let Some(new_capacity) = self.capacity.checked_add(bytes) else {
            // u64 无法表达的 arena 直接封顶，使任意现实设备预算 fail-closed，且不伪造重叠 offset。
            self.capacity = u64::MAX;
            return None;
        };
        self.capacity = new_capacity;
        Some(offset)
    }

    fn release(&mut self, offset: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut merged_offset = offset;
        let mut merged_bytes = bytes;

        let previous = self
            .free_by_offset
            .range(..offset)
            .next_back()
            .map(|(block_offset, block_bytes)| (*block_offset, *block_bytes));
        if let Some((block_offset, block_bytes)) = previous {
            if block_offset.checked_add(block_bytes) == Some(offset) {
                self.remove_free_block(block_offset, block_bytes);
                merged_offset = block_offset;
                merged_bytes = merged_bytes.saturating_add(block_bytes);
            }
        }

        let next = self
            .free_by_offset
            .range((Excluded(merged_offset), Unbounded))
            .next()
            .map(|(block_offset, block_bytes)| (*block_offset, *block_bytes));
        if let Some((block_offset, block_bytes)) = next {
            if merged_offset.checked_add(merged_bytes) == Some(block_offset) {
                self.remove_free_block(block_offset, block_bytes);
                merged_bytes = merged_bytes.saturating_add(block_bytes);
            }
        }
        self.insert_free_block(merged_offset, merged_bytes);
    }

    fn insert_free_block(&mut self, offset: u64, bytes: u64) {
        let previous = self.free_by_offset.insert(offset, bytes);
        let inserted = self.free_by_size.insert((bytes, offset));
        debug_assert!(previous.is_none() && inserted, "free-list indexes diverged");
    }

    fn remove_free_block(&mut self, offset: u64, bytes: u64) {
        let removed_offset = self.free_by_offset.remove(&offset);
        let removed_size = self.free_by_size.remove(&(bytes, offset));
        debug_assert_eq!(removed_offset, Some(bytes), "free-list offset index diverged");
        debug_assert!(removed_size, "free-list size index diverged");
    }
}

struct Planner<'a> {
    model: &'a NormalizedModel,
    tensors: Vec<&'a TensorInfo>,
    tensor_indexes: HashMap<&'a str, usize>,
    facts: Vec<TensorFact>,
    graph_inputs: Vec<bool>,
    graph_outputs: Vec<bool>,
    remaining_consumers: Vec<u64>,
    active: Vec<Option<ActiveTensor>>,
    slots: Vec<SlotState>,
    arena: Arena,
    alignment: u64,
    graph_end_event: u64,
    logical_live_bytes: u64,
    physical_live_bytes: u64,
    logical_peak_bytes: u64,
    planned_peak_bytes: u64,
    peak_event_id: u64,
    peak_event: String,
    peak_node_index: Option<u64>,
    peak_node_name: Option<String>,
    peak_op_type: Option<String>,
    peak_live_allocated_bytes: u64,
    peak_workspace_bytes: u64,
    peak_fragmentation_bytes: u64,
    inplace_reuse_count: u64,
    inplace_avoided_allocation_bytes: u64,
    planning_overflowed: bool,
    bounded_tensors: BTreeSet<usize>,
    unresolved_tensors: BTreeSet<usize>,
    trace: Vec<MemoryAllocationTrace>,
}

impl<'a> Planner<'a> {
    fn new(
        model: &'a NormalizedModel,
        symbol_bounds: &BTreeMap<String, u64>,
        alignment: u64,
    ) -> Self {
        let alignment = alignment.max(1);
        let tensors = model.tensors.values().collect::<Vec<_>>();
        let tensor_indexes = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let facts = tensors
            .iter()
            .map(|tensor| tensor_fact(tensor, symbol_bounds, alignment))
            .collect::<Vec<_>>();
        let mut graph_inputs = vec![false; tensors.len()];
        let mut graph_outputs = vec![false; tensors.len()];
        for name in &model.inputs {
            if let Some(index) = tensor_indexes.get(name.as_str()) {
                graph_inputs[*index] = true;
            }
        }
        for name in &model.outputs {
            if let Some(index) = tensor_indexes.get(name.as_str()) {
                graph_outputs[*index] = true;
            }
        }
        let mut remaining_consumers = vec![0_u64; tensors.len()];
        for node in &model.nodes {
            for name in node.inputs.iter().filter(|name| !name.is_empty()) {
                if let Some(index) = tensor_indexes.get(name.as_str()) {
                    if !tensors[*index].initializer {
                        remaining_consumers[*index] =
                            remaining_consumers[*index].saturating_add(1);
                    }
                }
            }
        }

        Self {
            model,
            tensors,
            tensor_indexes,
            facts,
            graph_inputs,
            graph_outputs,
            remaining_consumers,
            active: std::iter::repeat_with(|| None)
                .take(model.tensors.len())
                .collect(),
            slots: Vec::new(),
            arena: Arena::default(),
            alignment,
            graph_end_event: model.nodes.len() as u64 + 1,
            logical_live_bytes: 0,
            physical_live_bytes: 0,
            logical_peak_bytes: 0,
            planned_peak_bytes: 0,
            peak_event_id: 0,
            peak_event: "graph_start".to_string(),
            peak_node_index: None,
            peak_node_name: None,
            peak_op_type: None,
            peak_live_allocated_bytes: 0,
            peak_workspace_bytes: 0,
            peak_fragmentation_bytes: 0,
            inplace_reuse_count: 0,
            inplace_avoided_allocation_bytes: 0,
            planning_overflowed: false,
            bounded_tensors: BTreeSet::new(),
            unresolved_tensors: BTreeSet::new(),
            trace: Vec::new(),
        }
    }

    fn tensor_index(&self, name: &str) -> Option<usize> {
        self.tensor_indexes.get(name).copied()
    }

    fn record_size_evidence(&mut self, tensor_index: usize) {
        if self.facts[tensor_index].used_profile_bound {
            self.bounded_tensors.insert(tensor_index);
        }
        if self.facts[tensor_index].logical_bytes.is_none() {
            self.unresolved_tensors.insert(tensor_index);
        }
        if self.facts[tensor_index].alignment_overflowed {
            self.planning_overflowed = true;
            self.arena.capacity = u64::MAX;
        }
    }

    fn activate_regular(&mut self, tensor_index: usize, event: u64) {
        if self.tensors[tensor_index].initializer || self.active[tensor_index].is_some() {
            return;
        }
        self.record_size_evidence(tensor_index);
        let fact = self.facts[tensor_index].clone();
        let mut slot_id = None;
        let mut arena_offset = None;
        if let Some(allocated_bytes) = fact.allocated_bytes {
            if allocated_bytes > 0 {
                if let Some(offset) = self.arena.allocate(allocated_bytes) {
                    let new_slot_id = self.slots.len();
                    self.slots.push(SlotState {
                        offset,
                        capacity: allocated_bytes,
                        reference_count: 1,
                    });
                    slot_id = Some(new_slot_id);
                    arena_offset = Some(offset);
                    self.physical_live_bytes = self
                        .physical_live_bytes
                        .saturating_add(allocated_bytes);
                } else {
                    self.planning_overflowed = true;
                }
            }
        }
        let logical_bytes = fact.logical_bytes.unwrap_or(0);
        self.logical_live_bytes = match self.logical_live_bytes.checked_add(logical_bytes) {
            Some(bytes) => bytes,
            None => {
                self.planning_overflowed = true;
                u64::MAX
            }
        };
        let trace_index = self.trace.len();
        self.trace.push(MemoryAllocationTrace {
            kind: "tensor".to_string(),
            name: self.tensors[tensor_index].name.clone(),
            logical_bytes: fact.logical_bytes,
            allocated_bytes: fact.allocated_bytes,
            arena_offset,
            first_event: event,
            last_event: self.graph_end_event,
            alias_of: None,
            size_source: fact.size_source,
            graph_output: self.graph_outputs[tensor_index],
        });
        self.active[tensor_index] = Some(ActiveTensor {
            logical_bytes,
            slot_id,
            trace_index,
        });
    }

    fn activate_alias(
        &mut self,
        output_index: usize,
        source_index: usize,
        event: u64,
    ) -> bool {
        if output_index == source_index
            || self.tensors[output_index].initializer
            || self.active[output_index].is_some()
        {
            return false;
        }
        let Some(source_slot_id) = self.active[source_index]
            .as_ref()
            .and_then(|active| active.slot_id)
        else {
            return false;
        };
        let Some(requested_bytes) = self.facts[output_index].allocated_bytes else {
            return false;
        };
        let source_slot = &self.slots[source_slot_id];
        if requested_bytes == 0
            || source_slot.reference_count != 1
            || requested_bytes > source_slot.capacity
        {
            return false;
        }
        let slot_capacity = source_slot.capacity;
        let slot_offset = source_slot.offset;
        self.record_size_evidence(output_index);
        // 只有独占 slot 才能进入此分支，因此 alias 后引用数必为 2。
        self.slots[source_slot_id].reference_count = 2;
        let logical_bytes = self.facts[output_index].logical_bytes.unwrap_or(0);
        self.logical_live_bytes = match self.logical_live_bytes.checked_add(logical_bytes) {
            Some(bytes) => bytes,
            None => {
                self.planning_overflowed = true;
                u64::MAX
            }
        };
        let trace_index = self.trace.len();
        self.trace.push(MemoryAllocationTrace {
            kind: "tensor".to_string(),
            name: self.tensors[output_index].name.clone(),
            logical_bytes: self.facts[output_index].logical_bytes,
            allocated_bytes: Some(slot_capacity),
            arena_offset: Some(slot_offset),
            first_event: event,
            last_event: self.graph_end_event,
            alias_of: Some(self.tensors[source_index].name.clone()),
            size_source: self.facts[output_index].size_source.clone(),
            graph_output: self.graph_outputs[output_index],
        });
        self.active[output_index] = Some(ActiveTensor {
            logical_bytes,
            slot_id: Some(source_slot_id),
            trace_index,
        });
        self.inplace_reuse_count = self.inplace_reuse_count.saturating_add(1);
        self.inplace_avoided_allocation_bytes = self
            .inplace_avoided_allocation_bytes
            .saturating_add(requested_bytes);
        true
    }

    fn release_tensor(&mut self, tensor_index: usize, event: u64) {
        let Some(active) = self.active[tensor_index].take() else {
            return;
        };
        self.logical_live_bytes = self
            .logical_live_bytes
            .saturating_sub(active.logical_bytes);
        self.trace[active.trace_index].last_event = event;
        if let Some(slot_id) = active.slot_id {
            let slot = &mut self.slots[slot_id];
            slot.reference_count = slot
                .reference_count
                .checked_sub(1)
                .expect("active tensor must own one slot reference");
            if slot.reference_count == 0 {
                self.physical_live_bytes = self
                    .physical_live_bytes
                    .checked_sub(slot.capacity)
                    .expect("live slot bytes must include the released capacity");
                self.arena.release(slot.offset, slot.capacity);
            }
        }
    }

    fn sample_peak(
        &mut self,
        event_id: u64,
        event: &str,
        node_index: Option<usize>,
        workspace_bytes: u64,
    ) {
        self.logical_peak_bytes = self.logical_peak_bytes.max(self.logical_live_bytes);
        if self.arena.capacity <= self.planned_peak_bytes {
            return;
        }
        self.planned_peak_bytes = self.arena.capacity;
        self.peak_event_id = event_id;
        self.peak_event = event.to_string();
        self.peak_node_index = node_index.map(|index| index as u64);
        self.peak_node_name = node_index.and_then(|index| self.model.nodes[index].name.clone());
        self.peak_op_type = node_index.map(|index| self.model.nodes[index].op_type.clone());
        self.peak_live_allocated_bytes = self.physical_live_bytes;
        self.peak_workspace_bytes = workspace_bytes;
        self.peak_fragmentation_bytes = self.arena.capacity.saturating_sub(
            self.physical_live_bytes
                .saturating_add(workspace_bytes),
        );
    }

    fn allocate_workspace(
        &mut self,
        node_index: usize,
        event: u64,
        logical_bytes: u64,
    ) -> (u64, Option<u64>) {
        let aligned_bytes = align_up(logical_bytes, self.alignment);
        if aligned_bytes == Some(0) {
            return (0, None);
        }
        let allocated_bytes = aligned_bytes.unwrap_or(u64::MAX);
        if aligned_bytes.is_none() {
            self.planning_overflowed = true;
            self.arena.capacity = u64::MAX;
        }
        let offset = aligned_bytes.and_then(|bytes| self.arena.allocate(bytes));
        if aligned_bytes.is_some() && offset.is_none() {
            self.planning_overflowed = true;
        }
        let node = &self.model.nodes[node_index];
        let name = node.name.clone().unwrap_or_else(|| {
            format!("node#{node_index}:{}::{}", node.domain, node.op_type)
        });
        self.trace.push(MemoryAllocationTrace {
            kind: "workspace".to_string(),
            name,
            logical_bytes: Some(logical_bytes),
            allocated_bytes: aligned_bytes,
            arena_offset: offset,
            first_event: event,
            last_event: event,
            alias_of: None,
            size_source: if aligned_bytes.is_some() {
                "profile_workspace"
            } else {
                "allocation_overflow"
            }
            .to_string(),
            graph_output: false,
        });
        (allocated_bytes, offset)
    }

    fn finish(mut self) -> ActivationMemoryPlan {
        let confidence = if self.planning_overflowed
            || self.model.shape_inference_status == "failed"
            || (self.logical_peak_bytes == 0 && !self.unresolved_tensors.is_empty())
        {
            "low"
        } else if self.unresolved_tensors.is_empty() && self.bounded_tensors.is_empty() {
            "high"
        } else {
            "medium"
        }
        .to_string();
        // trace 按执行事件与模型声明顺序生成，无需为稳定输出再做全量排序。
        let peak_contributors = peak_contributors(&self.trace, self.peak_event_id);

        ActivationMemoryPlan {
            logical_peak_bytes: self.logical_peak_bytes,
            planned_arena_bytes: self.planned_peak_bytes,
            alignment_bytes: self.alignment,
            algorithm: PLANNER_ALGORITHM.to_string(),
            planning_overflowed: self.planning_overflowed,
            peak_event: self.peak_event,
            peak_node_index: self.peak_node_index,
            peak_node_name: self.peak_node_name,
            peak_op_type: self.peak_op_type,
            peak_live_allocated_bytes: self.peak_live_allocated_bytes,
            peak_workspace_bytes: self.peak_workspace_bytes,
            peak_fragmentation_bytes: self.peak_fragmentation_bytes,
            inplace_reuse_count: self.inplace_reuse_count,
            inplace_avoided_allocation_bytes: self.inplace_avoided_allocation_bytes,
            peak_contributors,
            allocation_trace: self.trace,
            confidence,
            bounded_dynamic_tensor_count: self.bounded_tensors.len() as u64,
            unresolved_tensor_size_count: self.unresolved_tensors.len() as u64,
        }
    }
}

pub(crate) fn plan_activation_memory(
    model: &NormalizedModel,
    profile: &TargetProfile,
) -> ActivationMemoryPlan {
    plan_with_contract(
        model,
        &profile.symbol_bounds,
        profile.tensor_alignment_bytes,
        Some(profile),
    )
}

pub(crate) fn plan_activation_memory_with_defaults(
    model: &NormalizedModel,
    symbol_bounds: &BTreeMap<String, u64>,
) -> ActivationMemoryPlan {
    plan_with_contract(model, symbol_bounds, 1, None)
}

fn plan_with_contract(
    model: &NormalizedModel,
    symbol_bounds: &BTreeMap<String, u64>,
    alignment: u64,
    profile: Option<&TargetProfile>,
) -> ActivationMemoryPlan {
    let mut planner = Planner::new(model, symbol_bounds, alignment);
    let op_rules = profile.map(|profile| {
        profile
            .allowed_ops
            .iter()
            .map(|((domain, op_type), rule)| ((domain.as_str(), op_type.as_str()), rule))
            .collect::<HashMap<_, _>>()
    });

    let graph_input_indexes = model
        .inputs
        .iter()
        .filter_map(|name| planner.tensor_index(name))
        .collect::<Vec<_>>();
    for tensor_index in &graph_input_indexes {
        planner.activate_regular(*tensor_index, 0);
    }
    planner.sample_peak(0, "graph_start", None, 0);
    for tensor_index in graph_input_indexes {
        if planner.remaining_consumers[tensor_index] == 0
            && !planner.graph_outputs[tensor_index]
        {
            planner.release_tensor(tensor_index, 0);
        }
    }

    for (node_index, node) in model.nodes.iter().enumerate() {
        let event = node_index as u64 + 1;
        let input_indexes = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .filter_map(|name| planner.tensor_index(name))
            .collect::<Vec<_>>();
        for tensor_index in &input_indexes {
            planner.activate_regular(*tensor_index, event);
        }

        let rule = op_rules.as_ref().and_then(|rules| {
            rules
                .get(&(node.domain.as_str(), node.op_type.as_str()))
                .copied()
        });
        let mut aliased_output = None;
        if let (Some(rule), Some(output_name)) = (rule, node.outputs.first()) {
            if let Some(input_position) = rule.first_output_inplace_input_index {
                if let (Some(source_name), Some(output_index)) =
                    (node.inputs.get(input_position), planner.tensor_index(output_name))
                {
                    if let Some(source_index) = planner.tensor_index(source_name) {
                        let current_node_uses = input_indexes
                            .iter()
                            .filter(|index| **index == source_index)
                            .count() as u64;
                        let source_is_safe = !planner.tensors[source_index].initializer
                            && !planner.graph_inputs[source_index]
                            && !planner.graph_outputs[source_index]
                            && current_node_uses > 0
                            && planner.remaining_consumers[source_index] == current_node_uses;
                        if source_is_safe
                            && planner.activate_alias(output_index, source_index, event)
                        {
                            aliased_output = Some(output_index);
                        }
                    }
                }
            }
        }
        let output_indexes = node
            .outputs
            .iter()
            .filter(|name| !name.is_empty())
            .filter_map(|name| planner.tensor_index(name))
            .collect::<Vec<_>>();
        for tensor_index in &output_indexes {
            if Some(*tensor_index) != aliased_output {
                planner.activate_regular(*tensor_index, event);
            }
        }

        let workspace_logical_bytes = rule.map(|rule| rule.workspace_bytes).unwrap_or(0);
        let (workspace_bytes, workspace_offset) =
            planner.allocate_workspace(node_index, event, workspace_logical_bytes);
        planner.sample_peak(event, "node", Some(node_index), workspace_bytes);

        for tensor_index in input_indexes {
            planner.remaining_consumers[tensor_index] =
                planner.remaining_consumers[tensor_index].saturating_sub(1);
            if planner.remaining_consumers[tensor_index] == 0
                && !planner.graph_outputs[tensor_index]
            {
                planner.release_tensor(tensor_index, event);
            }
        }
        for tensor_index in output_indexes {
            if planner.remaining_consumers[tensor_index] == 0
                && !planner.graph_outputs[tensor_index]
            {
                planner.release_tensor(tensor_index, event);
            }
        }
        if let Some(offset) = workspace_offset {
            planner.arena.release(offset, workspace_bytes);
        }
    }

    let graph_output_indexes = model
        .outputs
        .iter()
        .filter_map(|name| planner.tensor_index(name))
        .collect::<Vec<_>>();
    let graph_end_event = planner.graph_end_event;
    for tensor_index in graph_output_indexes {
        planner.activate_regular(tensor_index, graph_end_event);
    }
    planner.sample_peak(graph_end_event, "graph_end", None, 0);
    planner.finish()
}

fn tensor_fact(
    tensor: &TensorInfo,
    symbol_bounds: &BTreeMap<String, u64>,
    alignment: u64,
) -> TensorFact {
    let bounded_size = tensor.byte_size_with_bounds(symbol_bounds);
    let (logical_bytes, used_profile_bound) = bounded_size
        .map(|(bytes, used_bound)| (Some(bytes), used_bound))
        .unwrap_or((None, false));
    let allocated_bytes = logical_bytes.and_then(|bytes| align_up(bytes, alignment));
    let alignment_overflowed = logical_bytes.is_some() && allocated_bytes.is_none();
    let size_source = if alignment_overflowed {
        "allocation_overflow"
    } else if tensor.bytes.is_some() {
        "declared"
    } else if tensor.static_shape_byte_size().is_some() {
        "static_shape"
    } else if used_profile_bound {
        "profile_bound"
    } else {
        "unresolved"
    }
    .to_string();
    TensorFact {
        logical_bytes,
        allocated_bytes,
        size_source,
        used_profile_bound,
        alignment_overflowed,
    }
}

fn align_up(bytes: u64, alignment: u64) -> Option<u64> {
    let alignment = alignment.max(1);
    let remainder = bytes % alignment;
    if remainder == 0 {
        Some(bytes)
    } else {
        bytes.checked_add(alignment - remainder)
    }
}

fn peak_contributors(
    trace: &[MemoryAllocationTrace],
    peak_event: u64,
) -> Vec<PeakMemoryContributor> {
    let mut by_offset = HashMap::<u64, (&MemoryAllocationTrace, u64)>::new();
    for item in trace.iter().filter(|item| {
        item.kind == "tensor"
            && item.first_event <= peak_event
            && peak_event <= item.last_event
            && item.allocated_bytes.unwrap_or(0) > 0
            && item.arena_offset.is_some()
    }) {
        let offset = item.arena_offset.unwrap_or(0);
        let bytes = item.allocated_bytes.unwrap_or(0);
        let should_replace = match by_offset.get(&offset) {
            None => true,
            Some((current, _)) => {
                item.first_event > current.first_event
                    || (item.first_event == current.first_event
                        && item.alias_of.is_some()
                        && current.alias_of.is_none())
                    || (item.first_event == current.first_event
                        && item.alias_of.is_some() == current.alias_of.is_some()
                        && item.name < current.name)
            }
        };
        if should_replace {
            by_offset.insert(offset, (item, bytes));
        }
    }
    let mut contributors = Vec::with_capacity(TOP_CONTRIBUTOR_LIMIT);
    for (item, bytes) in by_offset.into_values() {
        contributors.push(PeakMemoryContributor {
            tensor: item.name.clone(),
            allocated_bytes: bytes,
        });
        contributors.sort_by(contributor_order);
        if contributors.len() > TOP_CONTRIBUTOR_LIMIT {
            contributors.pop();
        }
    }
    contributors
}

fn contributor_order(
    left: &PeakMemoryContributor,
    right: &PeakMemoryContributor,
) -> std::cmp::Ordering {
    right
        .allocated_bytes
        .cmp(&left.allocated_bytes)
        .then_with(|| left.tensor.cmp(&right.tensor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_coalesces_blocks_and_reuses_best_fit() {
        let mut arena = Arena::default();
        let first = arena.allocate(16).unwrap();
        let middle = arena.allocate(32).unwrap();
        let last = arena.allocate(16).unwrap();
        arena.release(first, 16);
        arena.release(last, 16);
        arena.release(middle, 32);

        assert_eq!(arena.allocate(48), Some(0));
        assert_eq!(arena.capacity, 64);
        assert_eq!(arena.free_by_offset.get(&48), Some(&16));
        assert!(arena.free_by_size.contains(&(16, 48)));
    }

    #[test]
    fn arena_overflow_fails_closed_without_reusing_the_same_offset() {
        let mut arena = Arena::default();

        assert_eq!(arena.allocate(u64::MAX), Some(0));
        assert_eq!(arena.allocate(1), None);
        assert_eq!(arena.capacity, u64::MAX);
    }

    #[test]
    fn alignment_overflow_is_not_rounded_down() {
        assert_eq!(align_up(u64::MAX - 1, 4), None);
    }
}
