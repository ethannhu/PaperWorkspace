"""Semantic, hardware-aware graph partitioning.

The partitioner follows the design in ``docs/partition.md``:

1. label nodes with a small semantic feature vector;
2. form local stage blocks from compatible motifs;
3. greedily merge blocks when communication and locality gains outweigh the
   loss of parallelism, then rebuild the partition DAG.

It deliberately returns the same lightweight object shape used by the
framework scheduler.  Keeping the algorithm here makes it independently
testable without coupling its scoring rules to scheduling code.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .demo_framework import GraphFeatures, Partition


COMPUTE_ROLES = {"DENSE_COMPUTE", "COMPUTE"}
REDUCTION_ROLES = {"REDUCTION"}
ELEMENTWISE_ROLES = {"ELEMENTWISE"}
COMMUNICATION_ROLES = {"COMMUNICATION"}
MEMORY_ROLES = {"MEMORY"}


@dataclass(frozen=True)
class NodeFeature:
    id: int
    op: str
    pipe: str
    cycles: int
    role: str
    indegree: int
    outdegree: int
    input_tensors: tuple[int, ...]
    output_tensors: tuple[int, ...]
    criticality: float
    boundary_score: float


@dataclass
class SemanticBlock:
    id: int
    nodes: list[int]
    semantic_type: str
    cycles: int
    pipe_cycles: dict[str, int] = field(default_factory=dict)
    input_tensors: set[int] = field(default_factory=set)
    output_tensors: set[int] = field(default_factory=set)
    internal_comm: float = 0.0
    external_comm: float = 0.0
    buffer_pressure: float = 0.0
    pipelineability: float = 0.0
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)


def _role(op: str, current: str) -> str:
    """Map the framework's legacy labels to the design's semantic roles."""
    if op in {"ALLOC", "FREE"}:
        return "MEMORY"
    if op in {"COPY_IN", "COPY_OUT", "MOVE"} or current == "COMMUNICATION":
        return "COMMUNICATION"
    if current == "DENSE_COMPUTE":
        return "COMPUTE"
    if current == "REDUCTION":
        return "REDUCTION"
    if current == "ELEMENTWISE":
        return "ELEMENTWISE"
    return "OTHER"


def extract_features(features: GraphFeatures) -> dict[int, NodeFeature]:
    """Create the semantic feature vector consumed by the matcher/scorer."""
    result: dict[int, NodeFeature] = {}
    for node in features.topo_order:
        op_data = features.op_by_id[node]
        op = str(op_data.get("op", ""))
        pipe = str(op_data.get("pipe", "UNKNOWN"))
        cycles = int(op_data.get("cycles", 0))
        indegree = len(features.preds[node])
        outdegree = len(features.succs[node])
        resource_switch = sum(
            1 for successor in features.succs[node]
            if str(features.op_by_id[successor].get("pipe", "UNKNOWN")) != pipe
        )
        # Fan-in/out are intentionally stronger than a pipe change: they are
        # the most reliable structural signal that a merge would remove
        # parallelism.
        boundary_score = (
            2.0 * max(0, indegree - 1)
            + 2.0 * max(0, outdegree - 1)
            + 0.5 * resource_switch
        )
        result[node] = NodeFeature(
            id=node,
            op=op,
            pipe=pipe,
            cycles=cycles,
            role=_role(op, features.semantic[node]),
            indegree=indegree,
            outdegree=outdegree,
            input_tensors=tuple(sorted(features.input_tensors.get(node, set()))),
            output_tensors=tuple(sorted(features.output_tensors.get(node, set()))),
            criticality=float(features.rank_u[node]),
            boundary_score=boundary_score,
        )
    return result


def _initial_block(feature: NodeFeature) -> SemanticBlock:
    return SemanticBlock(
        id=feature.id,
        nodes=[feature.id],
        semantic_type=feature.role,
        cycles=feature.cycles,
        pipe_cycles={feature.pipe: feature.cycles},
        input_tensors=set(feature.input_tensors),
        output_tensors=set(feature.output_tensors),
        pipelineability=1.0,
    )


def _motif_type(left: NodeFeature, right: NodeFeature) -> str:
    roles = {left.role, right.role}
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return "ComputeFusionBlock"
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return "ReductionStage"
    if left.role == "COMMUNICATION" or right.role == "COMMUNICATION":
        return "CommunicationComputeStage"
    if roles == {"COMPUTE"} and left.pipe == right.pipe:
        return "ComputeTile"
    if roles == {"ELEMENTWISE"}:
        return "ElementwiseChain"
    return "MixedStage"


def _semantic_affinity(left: NodeFeature, right: NodeFeature) -> float:
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return 4.0
    if left.role == right.role == "ELEMENTWISE":
        return 2.5
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return 2.5
    if left.pipe == right.pipe:
        return 1.0
    if {left.pipe, right.pipe} == {"PIPE_MTE2", "PIPE_M"}:
        return 1.5
    return 0.0


def _explicit_stage_pair(left: NodeFeature, right: NodeFeature) -> bool:
    """Return whether a cross-resource edge is an intentional stage boundary."""
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return True
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return True
    if left.role == "COMMUNICATION" and right.role == "COMPUTE":
        return True
    # Keep the hardware-oriented MTE -> Cube -> MTE motif available for graphs
    # that expose MOVE/COPY-like stages as ordinary operations.
    return (
        left.pipe in {"PIPE_MTE2", "PIPE_MTE3"} and right.pipe == "PIPE_M"
    ) or (
        left.pipe == "PIPE_M" and right.pipe in {"PIPE_MTE2", "PIPE_MTE3"}
    )


def _merge_score(
    left: SemanticBlock,
    right: SemanticBlock,
    features: dict[int, NodeFeature],
    edge_bytes: int,
    max_ops: int,
    max_cycles: int,
) -> float:
    if len(left.nodes) + len(right.nodes) > max_ops:
        return float("-inf")
    if left.cycles + right.cycles > max_cycles:
        return float("-inf")
    tail = features[left.nodes[-1]]
    head = features[right.nodes[0]]
    # An elementwise producer normally feeds many independent compute tiles.
    # Fusing it into the first tile serializes that producer with the cube
    # stage and can move the tile to the end of the task priority order.
    if tail.role == "ELEMENTWISE" and head.role == "COMPUTE":
        return float("-inf")
    if tail.pipe != head.pipe and not _explicit_stage_pair(tail, head):
        return float("-inf")
    # The byte term is normalized so a large tensor strongly favors avoiding a
    # writeback, while tiny edges still need semantic affinity to merge.
    communication_saved = min(6.0, edge_bytes / 1024.0)
    locality_gain = 1.5 if left.output_tensors & right.input_tensors else 0.0
    resource_diversity = 0.8 if tail.pipe != head.pipe else -0.4
    parallelism_loss = 2.5 * (tail.outdegree > 1 or head.indegree > 1)
    boundary_penalty = tail.boundary_score + head.boundary_score
    serial_penalty = 0.5 if tail.role == head.role == "COMPUTE" else 0.0
    return (
        communication_saved
        + locality_gain
        + _semantic_affinity(tail, head)
        + resource_diversity
        - parallelism_loss
        - boundary_penalty
        - serial_penalty
    )


def _merge(left: SemanticBlock, right: SemanticBlock, motif: str) -> None:
    left.nodes.extend(right.nodes)
    left.semantic_type = motif if left.semantic_type != motif else left.semantic_type
    left.cycles += right.cycles
    for pipe, cycles in right.pipe_cycles.items():
        left.pipe_cycles[pipe] = left.pipe_cycles.get(pipe, 0) + cycles
    left.input_tensors |= right.input_tensors
    left.output_tensors |= right.output_tensors
    left.internal_comm += right.internal_comm
    left.buffer_pressure += right.buffer_pressure
    left.pipelineability = max(left.pipelineability, right.pipelineability)


def _build_stage_blocks(features: GraphFeatures) -> tuple[dict[int, SemanticBlock], dict[int, NodeFeature]]:
    node_features = extract_features(features)
    blocks = {node: _initial_block(feature) for node, feature in node_features.items()}
    return blocks, node_features


def _merge_blocks(
    features: GraphFeatures,
    blocks: dict[int, SemanticBlock],
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> None:
    """Merge the highest-scoring live edge, updating only its neighborhood."""
    owner = {node: node for node in node_features}
    heap: list[tuple[float, int, int, int]] = []
    serial = 0

    def push(source: int, target: int) -> None:
        nonlocal serial
        left_id, right_id = owner[source], owner[target]
        if left_id == right_id or left_id not in blocks or right_id not in blocks:
            return
        left, right = blocks[left_id], blocks[right_id]
        # A merge must be a convex, linear stage connection.  In particular,
        # do not absorb a branch or a join: contracting either can introduce a
        # cycle in the block DAG even when the operation graph is acyclic.
        if source != left.nodes[-1] or target != right.nodes[0]:
            return
        if any(owner[pred] != left_id for pred in features.preds[target]):
            return
        if any(owner[successor] != right_id for successor in features.succs[source]):
            return
        score = _merge_score(
            left,
            right,
            node_features,
            features.edge_sizes.get((source, target), 0),
            max_ops,
            max_cycles,
        )
        heapq.heappush(heap, (-score, serial, left_id, right_id))
        serial += 1

    for source in features.topo_order:
        for target in sorted(features.succs[source]):
            push(source, target)

    while heap:
        neg_score, _, left_id, right_id = heapq.heappop(heap)
        if left_id not in blocks or right_id not in blocks:
            continue
        left, right = blocks[left_id], blocks[right_id]
        if owner[left.nodes[-1]] != left_id or owner[right.nodes[0]] != right_id:
            continue
        score = -neg_score
        if score <= 0:
            break
        tail, head = left.nodes[-1], right.nodes[0]
        if right_id not in {owner[node] for node in features.succs[tail]}:
            continue
        _merge(left, right, _motif_type(node_features[tail], node_features[head]))
        for node in right.nodes:
            owner[node] = left_id
        del blocks[right_id]
        tail = left.nodes[-1]
        head = left.nodes[0]
        for successor in features.succs[tail]:
            push(tail, successor)
        for predecessor in features.preds[head]:
            push(predecessor, head)


def build_partition_dag(features: GraphFeatures, blocks: dict[int, SemanticBlock]) -> list[Partition]:
    """Convert semantic blocks into the scheduler's compact partition objects."""
    from .demo_framework import Partition

    ordered = sorted(blocks.values(), key=lambda block: (min(block.nodes), block.id))
    remap = {block.id: index for index, block in enumerate(ordered)}
    owner = {node: remap[block.id] for block in ordered for node in block.nodes}
    partitions: list[Partition] = []
    for index, block in enumerate(ordered):
        partitions.append(
            Partition(
                id=index,
                ops=list(block.nodes),
                cycles=block.cycles,
                rank_u=max(features.rank_u[node] for node in block.nodes),
                pipe_cycles=dict(block.pipe_cycles),
            )
        )
    for source in features.topo_order:
        for target in features.succs[source]:
            source_id, target_id = owner[source], owner[target]
            if source_id != target_id:
                partitions[source_id].succs.add(target_id)
                partitions[target_id].preds.add(source_id)
    return partitions


def semantic_partition(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
) -> list[Partition]:
    """Partition an analyzed graph using semantic motifs and merge scores."""
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")
    blocks, node_features = _build_stage_blocks(features)
    _merge_blocks(features, blocks, node_features, max_ops, max_cycles)
    return build_partition_dag(features, blocks)


__all__ = [
    "NodeFeature",
    "SemanticBlock",
    "extract_features",
    "semantic_partition",
]
