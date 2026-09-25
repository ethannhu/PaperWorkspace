"""Semantic, hardware-aware graph partitioning.

The partitioner follows the design in ``docs/partition.md``:

1. label nodes with a small semantic feature vector;
2. form local stage blocks from compatible motifs;
3. greedily merge blocks when communication and locality gains outweigh the
   loss of parallelism, then rebuild the partition DAG.

It deliberately returns the lightweight ``Partition`` objects consumed by the
current family strategies.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm_common import GraphFeatures, Partition


COMPUTE_ROLES = {"DENSE_COMPUTE", "COMPUTE"}
REDUCTION_ROLES = {"REDUCTION"}
ELEMENTWISE_ROLES = {"ELEMENTWISE"}
COMMUNICATION_ROLES = {"COMMUNICATION"}
MEMORY_ROLES = {"MEMORY"}

# These are the fixed evaluator settings in ``artifacts/data/config.txt``.
# The merge score is deliberately expressed in a small, dimensionless range,
# but its input is still the evaluator's transfer time rather than raw bytes.
_EVALUATOR_BANDWIDTH_BYTES_PER_CYCLE = 60
_COMMUNICATION_REFERENCE_CYCLES = 16


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
    # Price the avoided partition boundary by its DMA time.  Raw byte scores
    # made the result depend on an arbitrary KiB unit and saturated every
    # tensor above 6 KiB at the same reward.  The logarithm keeps this term on
    # the scale of the semantic terms while preserving the 4 KiB -> 32 KiB
    # distinction present in the evaluator (69 -> 547 transfer cycles).
    communication_saved = _communication_saved(edge_bytes)
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


def _owner_map(blocks: dict[int, SemanticBlock]) -> dict[int, int]:
    return {node: block.id for block in blocks.values() for node in block.nodes}


def _edges_between(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
) -> list[tuple[int, int]]:
    right_nodes = set(right.nodes)
    return [
        (source, target)
        for source in left.nodes
        for target in features.succs[source]
        if target in right_nodes
    ]


def _crossing_bytes(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
    edge_pairs: list[tuple[int, int]] | None = None,
) -> int:
    """Return tensor bytes materialized at a block boundary exactly once.

    A tensor may feed several operations in ``right``.  Summing operation
    edges counts that one DMA repeatedly, while the evaluator transfers it
    once per destination task.  COPY contraction can hide tensor identity;
    retain the contracted-edge estimate as a fallback for that case.
    """
    produced = set().union(
        *(features.output_tensors.get(node, set()) for node in left.nodes)
    )
    consumed = set().union(
        *(features.input_tensors.get(node, set()) for node in right.nodes)
    )
    shared_tensors = produced & consumed
    if shared_tensors:
        return sum(
            int(features.tensor_by_id[tensor_id].get("size", 0))
            for tensor_id in shared_tensors
        )
    if edge_pairs is None:
        edge_pairs = _edges_between(features, left, right)
    return sum(features.edge_sizes.get(edge, 0) for edge in edge_pairs)


def _communication_saved(edge_bytes: int) -> float:
    """Normalize avoided transfer time for use in a merge score.

    ``ceil(bytes / 60)`` is the payload portion of the official evaluator's
    transfer model.  Fixed wait time is excluded: whether it applies depends
    on the later core placement and Q1/Q2 scenario, neither of which is known
    during scenario-independent semantic partitioning.  Sixteen cycles is a
    score-scale reference, not a hardware parameter; it keeps this reward
    comparable with the 1--4 point semantic affinity terms.
    """
    if edge_bytes <= 0:
        return 0.0
    transfer_cycles = math.ceil(edge_bytes / _EVALUATOR_BANDWIDTH_BYTES_PER_CYCLE)
    return math.log1p(transfer_cycles / _COMMUNICATION_REFERENCE_CYCLES)


def _would_create_partition_cycle(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
) -> bool:
    """Detect non-convex merges that would create a cycle in the block DAG."""
    candidate = set(left.nodes) | set(right.nodes)
    seen: set[int] = set()
    stack = [
        successor
        for node in candidate
        for successor in features.succs[node]
        if successor not in candidate
    ]
    while stack:
        node = stack.pop()
        if node in candidate:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(features.succs[node])
    return False


def _singleton_repair_score(
    left: SemanticBlock,
    right: SemanticBlock,
    graph_features: GraphFeatures,
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> float:
    """Score a conservative repair merge after the strict motif pass.

    This pass is intentionally more permissive about fan-in/fan-out than
    ``_merge_blocks`` but still prices the parallelism loss.  It is meant to
    clean up isolated one-op blocks around otherwise obvious stage patterns.
    """
    if len(left.nodes) + len(right.nodes) > max_ops:
        return float("-inf")
    if left.cycles + right.cycles > max_cycles:
        return float("-inf")

    edge_pairs = _edges_between(graph_features, left, right)
    if not edge_pairs:
        return float("-inf")

    # Use the strongest touching edge as the semantic representative, while
    # communication reward accounts for every tensor crossing the block pair.
    source, target = max(
        edge_pairs,
        key=lambda edge: graph_features.edge_sizes.get(edge, 0),
    )
    tail = node_features[source]
    head = node_features[target]
    if tail.role == "ELEMENTWISE" and head.role == "COMPUTE":
        return float("-inf")
    if tail.pipe != head.pipe and not _explicit_stage_pair(tail, head):
        return float("-inf")
    if _would_create_partition_cycle(graph_features, left, right):
        return float("-inf")

    edge_bytes = _crossing_bytes(graph_features, left, right, edge_pairs)
    communication_saved = _communication_saved(edge_bytes)
    locality_gain = 1.5 if left.output_tensors & right.input_tensors else 0.0
    semantic_gain = max(
        _semantic_affinity(node_features[src], node_features[dst])
        for src, dst in edge_pairs
    )
    singleton_gain = 1.0 if len(left.nodes) == 1 or len(right.nodes) == 1 else 0.0
    resource_diversity = 0.8 if tail.pipe != head.pipe else -0.2

    # Fan-in/fan-out still matter, but this pass should not treat every join as
    # an absolute wall.  Large tensor reuse and strong semantics can buy through
    # a modest structural penalty.
    fan_loss = math.log2(1 + max(0, tail.outdegree - 1) + max(0, head.indegree - 1))
    boundary_penalty = 0.35 * (tail.boundary_score + head.boundary_score)
    serial_penalty = 0.5 if tail.role == head.role == "COMPUTE" else 0.0
    return (
        communication_saved
        + locality_gain
        + semantic_gain
        + singleton_gain
        + resource_diversity
        - fan_loss
        - boundary_penalty
        - serial_penalty
    )


def repair_singleton_blocks(
    features: GraphFeatures,
    blocks: dict[int, SemanticBlock],
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> None:
    """Merge profitable singleton blocks left by the strict semantic pass.

    This repair pass is intentionally separate from ``semantic_partition``'s
    default path.  It is useful as an experiment when isolated one-op blocks
    are too conservative, but it may also reduce parallelism around fan-in or
    fan-out structures.
    """
    for _ in range(2):
        owner = _owner_map(blocks)
        changed = False
        for node in features.topo_order:
            block_id = owner.get(node)
            if block_id not in blocks:
                continue
            block = blocks[block_id]
            if len(block.nodes) != 1:
                continue

            candidates: list[tuple[float, int, int, str]] = []
            for predecessor in sorted(features.preds[node]):
                left_id = owner.get(predecessor)
                if left_id is None or left_id == block_id or left_id not in blocks:
                    continue
                left = blocks[left_id]
                score = _singleton_repair_score(
                    left,
                    block,
                    features,
                    node_features,
                    max_ops,
                    max_cycles,
                )
                candidates.append((score, left_id, block_id, "pred"))
            for successor in sorted(features.succs[node]):
                right_id = owner.get(successor)
                if right_id is None or right_id == block_id or right_id not in blocks:
                    continue
                right = blocks[right_id]
                score = _singleton_repair_score(
                    block,
                    right,
                    features,
                    node_features,
                    max_ops,
                    max_cycles,
                )
                candidates.append((score, block_id, right_id, "succ"))
            if not candidates:
                continue

            score, left_id, right_id, _ = max(candidates, key=lambda item: item[0])
            if score <= 0 or left_id not in blocks or right_id not in blocks:
                continue

            left = blocks[left_id]
            right = blocks[right_id]
            edge_pairs = _edges_between(features, left, right)
            source, target = max(
                edge_pairs,
                key=lambda edge: features.edge_sizes.get(edge, 0),
            )
            _merge(left, right, _motif_type(node_features[source], node_features[target]))
            for merged_node in right.nodes:
                owner[merged_node] = left_id
            del blocks[right_id]
            changed = True

        if not changed:
            break


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
    """Convert semantic blocks into compact partition objects."""
    from .algorithm_common import Partition

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
    enable_singleton_repair: bool = False,
) -> list[Partition]:
    """Partition an analyzed graph using semantic motifs and merge scores."""
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")
    blocks, node_features = _build_stage_blocks(features)
    _merge_blocks(features, blocks, node_features, max_ops, max_cycles)
    if enable_singleton_repair:
        repair_singleton_blocks(features, blocks, node_features, max_ops, max_cycles)
    return build_partition_dag(features, blocks)


def semantic_partition_with_singleton_repair(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
) -> list[Partition]:
    """Partition with the optional singleton repair pass enabled."""
    return semantic_partition(
        features,
        max_ops=max_ops,
        max_cycles=max_cycles,
        enable_singleton_repair=True,
    )


__all__ = [
    "NodeFeature",
    "SemanticBlock",
    "extract_features",
    "repair_singleton_blocks",
    "semantic_partition",
    "semantic_partition_with_singleton_repair",
]
