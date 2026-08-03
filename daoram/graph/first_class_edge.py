"""Reference state machine for Grove's first-class-edge extension.

This module captures the logical position-maintenance protocol independently
from a concrete ORAM implementation.  It is intentionally small enough to use
as an executable specification while the Vertex Data, Vertex Meta, and Edge
Data ORAMs are integrated into :mod:`daoram.graph.grove`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


class AdjacencyRole(str, Enum):
    """The role of a vertex in an adjacency slot."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    UNDIRECTED = "undirected"
    STRUCTURAL = "structural"


@dataclass
class AdjacencyEntry:
    """A direct reference to one neighbor and, optionally, one real edge."""

    neighbor_id: Any
    neighbor_path: int
    edge_id: Optional[Any]
    edge_path: Optional[int]
    role: AdjacencyRole
    edge_valid: bool = True


@dataclass
class VertexRecord:
    """A first-class vertex and its padded logical adjacency slots."""

    vertex_id: Any
    value: Any
    path: int
    adjacency: Dict[Any, AdjacencyEntry] = field(default_factory=dict)


@dataclass
class EdgeRecord:
    """A first-class edge with no physical references to its endpoints."""

    edge_id: Any
    source_id: Any
    target_id: Any
    value: Any
    path: int
    directed: bool = True


@dataclass
class AdjacencyUpdate:
    """One fixed-format delayed update for one adjacency entry.

    The two validity flags are encrypted in the real protocol.  A vertex-only
    access sets ``vertex_valid`` and leaves ``edge_valid`` false.  An edge-only
    access from one endpoint sets both flags in the same logical meta block.
    """

    target_id: Any
    neighbor_id: Any
    edge_id: Optional[Any]
    new_vertex_path: int = 0
    new_edge_path: int = 0
    vertex_valid: bool = False
    edge_valid: bool = False
    epoch: int = 0

    def merged_with(self, newer: "AdjacencyUpdate") -> "AdjacencyUpdate":
        """Merge two pending blocks while retaining each newest valid field."""
        if (self.target_id, self.neighbor_id) != (newer.target_id, newer.neighbor_id):
            raise ValueError("Cannot merge updates for different adjacency entries.")

        edge_id = newer.edge_id if newer.edge_valid else self.edge_id
        return AdjacencyUpdate(
            target_id=self.target_id,
            neighbor_id=self.neighbor_id,
            edge_id=edge_id,
            new_vertex_path=(
                newer.new_vertex_path if newer.vertex_valid else self.new_vertex_path
            ),
            new_edge_path=newer.new_edge_path if newer.edge_valid else self.new_edge_path,
            vertex_valid=self.vertex_valid or newer.vertex_valid,
            edge_valid=self.edge_valid or newer.edge_valid,
            epoch=max(self.epoch, newer.epoch),
        )

    def apply(self, vertex: VertexRecord) -> None:
        """Apply this update to its target vertex."""
        if vertex.vertex_id != self.target_id:
            raise ValueError("Adjacency update applied to the wrong vertex.")
        if self.neighbor_id not in vertex.adjacency:
            raise KeyError("Adjacency update targets a missing entry.")

        entry = vertex.adjacency[self.neighbor_id]
        if self.vertex_valid:
            entry.neighbor_path = self.new_vertex_path
        if self.edge_valid:
            if not entry.edge_valid or entry.edge_id != self.edge_id:
                raise ValueError("Edge update does not match the target adjacency entry.")
            entry.edge_path = self.new_edge_path


class FirstClassEdgeState:
    """Executable specification of the confirmed first-class-edge design."""

    def __init__(
        self,
        max_degree: int,
        leaf_range: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        if max_degree <= 0:
            raise ValueError("max_degree must be positive.")
        if leaf_range <= 1:
            raise ValueError("leaf_range must be greater than one.")

        self.max_degree = max_degree
        self.leaf_range = leaf_range
        self.vertices: Dict[Any, VertexRecord] = {}
        self.edges: Dict[Any, EdgeRecord] = {}
        self._edge_by_pair: Dict[frozenset[Any], Any] = {}
        self._pending: Dict[Any, Dict[Any, AdjacencyUpdate]] = {}
        self._rng = rng or random.SystemRandom()
        self._epoch = 0
        self.last_schedule: Dict[str, int] = {}

    def _new_path(self) -> int:
        return self._rng.randrange(self.leaf_range)

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    @staticmethod
    def _pair_key(left: Any, right: Any) -> frozenset[Any]:
        if left == right:
            raise ValueError("Self-loops are outside the current simple-graph model.")
        return frozenset((left, right))

    def _require_vertex(self, vertex_id: Any) -> VertexRecord:
        if vertex_id not in self.vertices:
            raise KeyError(f"Unknown vertex: {vertex_id!r}")
        return self.vertices[vertex_id]

    def _require_edge(self, endpoint: Any, neighbor: Any) -> EdgeRecord:
        pair = self._pair_key(endpoint, neighbor)
        if pair not in self._edge_by_pair:
            raise KeyError(f"No edge between {endpoint!r} and {neighbor!r}.")
        return self.edges[self._edge_by_pair[pair]]

    def _enqueue(self, update: AdjacencyUpdate) -> None:
        pending_for_target = self._pending.setdefault(update.target_id, {})
        existing = pending_for_target.get(update.neighbor_id)
        pending_for_target[update.neighbor_id] = (
            update if existing is None else existing.merged_with(update)
        )

    def apply_pending_updates(self, vertex_id: Any) -> int:
        """Apply all pending adjacency updates without remapping the vertex.

        The concrete protocol performs this step after retrieving the vertex's
        Vertex Meta path.  Exposing it here also makes bulk initialization and
        invariant testing deterministic.
        """
        vertex = self._require_vertex(vertex_id)
        updates = self._pending.pop(vertex_id, {})
        for update in sorted(updates.values(), key=lambda item: item.epoch):
            update.apply(vertex)
        return len(updates)

    def apply_all_pending_updates(self) -> int:
        """Materialize all delayed updates, for setup and invariant checks."""
        total = 0
        for vertex_id in list(self._pending):
            total += self.apply_pending_updates(vertex_id)
        return total

    def pending_updates(self, vertex_id: Any) -> Dict[Any, AdjacencyUpdate]:
        """Return a shallow copy of the logical pending blocks for a vertex."""
        return dict(self._pending.get(vertex_id, {}))

    def add_vertex(self, vertex_id: Any, value: Any = None) -> VertexRecord:
        """Insert one isolated vertex with no real adjacency entries."""
        if vertex_id in self.vertices:
            raise ValueError(f"Duplicate vertex: {vertex_id!r}")
        vertex = VertexRecord(vertex_id=vertex_id, value=value, path=self._new_path())
        self.vertices[vertex_id] = vertex
        self.last_schedule = {"vertex_reads": 0, "edge_reads": 0, "meta_insertions": 0}
        return vertex

    def _check_degree_for_new_entry(self, vertex_id: Any) -> None:
        vertex = self._require_vertex(vertex_id)
        if len(vertex.adjacency) >= self.max_degree:
            raise OverflowError(f"Vertex {vertex_id!r} exceeds max_degree.")

    def _remap_local(
        self,
        local_vertex_ids: Iterable[Any],
        accessed_edge_ids: Iterable[Any] = (),
        edge_path_overrides: Optional[Mapping[Any, int]] = None,
    ) -> None:
        """Remap local records and directly repair or delay every reference."""
        local_ids = set(local_vertex_ids)
        accessed_edges = set(accessed_edge_ids)
        edge_path_overrides = dict(edge_path_overrides or {})
        unknown_overrides = set(edge_path_overrides) - accessed_edges
        if unknown_overrides:
            raise ValueError(
                f"Edge path overrides include unaccessed edges: {unknown_overrides!r}"
            )
        if any(not 0 <= path < self.leaf_range for path in edge_path_overrides.values()):
            raise ValueError("Edge path override is outside the configured leaf range.")
        new_vertex_paths = {vertex_id: self._new_path() for vertex_id in local_ids}
        new_edge_paths = {
            edge_id: edge_path_overrides.get(edge_id, self._new_path())
            for edge_id in accessed_edges
        }

        for vertex_id in local_ids:
            self.vertices[vertex_id].path = new_vertex_paths[vertex_id]
        for edge_id in accessed_edges:
            self.edges[edge_id].path = new_edge_paths[edge_id]

        real_insertions = 0
        for source_id in local_ids:
            source = self.vertices[source_id]
            for neighbor_id, entry in source.adjacency.items():
                edge_was_accessed = entry.edge_valid and entry.edge_id in accessed_edges
                if edge_was_accessed:
                    entry.edge_path = new_edge_paths[entry.edge_id]

                if neighbor_id in local_ids:
                    entry.neighbor_path = new_vertex_paths[neighbor_id]
                    reverse = self.vertices[neighbor_id].adjacency[source_id]
                    reverse.neighbor_path = new_vertex_paths[source_id]
                    if edge_was_accessed:
                        reverse.edge_path = new_edge_paths[entry.edge_id]
                    continue

                update = AdjacencyUpdate(
                    target_id=neighbor_id,
                    neighbor_id=source_id,
                    edge_id=entry.edge_id if edge_was_accessed else None,
                    new_vertex_path=new_vertex_paths[source_id],
                    new_edge_path=(new_edge_paths[entry.edge_id] if edge_was_accessed else 0),
                    vertex_valid=True,
                    edge_valid=edge_was_accessed,
                    epoch=self._next_epoch(),
                )
                self._enqueue(update)
                real_insertions += 1

        self.last_schedule = {
            "vertex_reads": len(local_ids),
            "edge_reads": len(accessed_edges),
            "meta_insertions": real_insertions,
        }

    def lookup_vertex(self, vertex_id: Any) -> VertexRecord:
        """Access and remap one vertex, delaying updates to its neighbors."""
        self.apply_pending_updates(vertex_id)
        self._remap_local((vertex_id,))
        return self.vertices[vertex_id]

    def insert_edge(
        self,
        source_id: Any,
        target_id: Any,
        value: Any = None,
        edge_id: Optional[Any] = None,
        directed: bool = True,
    ) -> EdgeRecord:
        """Insert one simple edge after retrieving both endpoint vertices."""
        pair = self._pair_key(source_id, target_id)
        if pair in self._edge_by_pair:
            raise ValueError("Parallel edges are outside the current model.")
        self._check_degree_for_new_entry(source_id)
        self._check_degree_for_new_entry(target_id)
        self.apply_pending_updates(source_id)
        self.apply_pending_updates(target_id)

        if edge_id is None:
            edge_id = (source_id, target_id)
        if edge_id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge_id!r}")

        edge = EdgeRecord(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            value=value,
            path=self._new_path(),
            directed=directed,
        )
        self.edges[edge_id] = edge
        self._edge_by_pair[pair] = edge_id

        source_role = AdjacencyRole.OUTGOING if directed else AdjacencyRole.UNDIRECTED
        target_role = AdjacencyRole.INCOMING if directed else AdjacencyRole.UNDIRECTED
        self.vertices[source_id].adjacency[target_id] = AdjacencyEntry(
            neighbor_id=target_id,
            neighbor_path=self.vertices[target_id].path,
            edge_id=edge_id,
            edge_path=edge.path,
            role=source_role,
        )
        self.vertices[target_id].adjacency[source_id] = AdjacencyEntry(
            neighbor_id=source_id,
            neighbor_path=self.vertices[source_id].path,
            edge_id=edge_id,
            edge_path=edge.path,
            role=target_role,
        )

        self._remap_local((source_id, target_id))
        return edge

    def insert_structural_link(self, left_id: Any, right_id: Any) -> None:
        """Insert a sparsification-only link without an Edge Data record."""
        pair = self._pair_key(left_id, right_id)
        if pair in self._edge_by_pair:
            raise ValueError("A real edge already occupies this vertex pair.")
        self._check_degree_for_new_entry(left_id)
        self._check_degree_for_new_entry(right_id)
        self.apply_pending_updates(left_id)
        self.apply_pending_updates(right_id)

        self.vertices[left_id].adjacency[right_id] = AdjacencyEntry(
            neighbor_id=right_id,
            neighbor_path=self.vertices[right_id].path,
            edge_id=None,
            edge_path=None,
            role=AdjacencyRole.STRUCTURAL,
            edge_valid=False,
        )
        self.vertices[right_id].adjacency[left_id] = AdjacencyEntry(
            neighbor_id=left_id,
            neighbor_path=self.vertices[left_id].path,
            edge_id=None,
            edge_path=None,
            role=AdjacencyRole.STRUCTURAL,
            edge_valid=False,
        )
        self._remap_local((left_id, right_id))

    def lookup_edge(
        self,
        endpoint_id: Any,
        neighbor_id: Any,
        include_neighbor: bool = False,
        new_edge_path: Optional[int] = None,
    ) -> EdgeRecord:
        """Access an edge through one endpoint, optionally with its neighbor."""
        self.apply_pending_updates(endpoint_id)
        edge = self._require_edge(endpoint_id, neighbor_id)
        local_ids = {endpoint_id}
        if include_neighbor:
            self.apply_pending_updates(neighbor_id)
            local_ids.add(neighbor_id)
        overrides = None if new_edge_path is None else {edge.edge_id: new_edge_path}
        self._remap_local(local_ids, (edge.edge_id,), edge_path_overrides=overrides)
        return edge

    def delete_edge(self, endpoint_id: Any, neighbor_id: Any) -> EdgeRecord:
        """Immediately remove one edge and both endpoint references."""
        self.apply_pending_updates(endpoint_id)
        self.apply_pending_updates(neighbor_id)
        edge = self._require_edge(endpoint_id, neighbor_id)

        del self.vertices[endpoint_id].adjacency[neighbor_id]
        del self.vertices[neighbor_id].adjacency[endpoint_id]
        del self._edge_by_pair[self._pair_key(endpoint_id, neighbor_id)]
        del self.edges[edge.edge_id]
        self._pending.get(endpoint_id, {}).pop(neighbor_id, None)
        self._pending.get(neighbor_id, {}).pop(endpoint_id, None)

        self._remap_local((endpoint_id, neighbor_id))
        self.last_schedule["edge_reads"] = 1
        return edge

    def delete_vertex(self, vertex_id: Any) -> VertexRecord:
        """Immediately remove a vertex, all incident edges, and all references."""
        self.apply_pending_updates(vertex_id)
        vertex = self._require_vertex(vertex_id)
        neighbor_ids = set(vertex.adjacency)
        for neighbor_id in neighbor_ids:
            self.apply_pending_updates(neighbor_id)

        local_survivors = set(neighbor_ids)
        for neighbor_id in neighbor_ids:
            entry = vertex.adjacency[neighbor_id]
            self.vertices[neighbor_id].adjacency.pop(vertex_id, None)
            self._pending.get(neighbor_id, {}).pop(vertex_id, None)
            if entry.edge_valid:
                edge = self.edges.pop(entry.edge_id)
                self._edge_by_pair.pop(self._pair_key(edge.source_id, edge.target_id), None)

        self._pending.pop(vertex_id, None)
        del self.vertices[vertex_id]
        if local_survivors:
            self._remap_local(local_survivors)
            real_insertions = self.last_schedule["meta_insertions"]
        else:
            real_insertions = 0
        self.last_schedule = {
            "vertex_reads": 1 + self.max_degree,
            "edge_reads": self.max_degree,
            "meta_insertions": real_insertions,
        }
        return vertex

    def neighbors(
        self,
        vertex_id: Any,
        role: Optional[AdjacencyRole] = None,
    ) -> list[Any]:
        """Return topology neighbors, optionally filtered by a local role bit."""
        self.apply_pending_updates(vertex_id)
        entries = self._require_vertex(vertex_id).adjacency.values()
        return [
            entry.neighbor_id
            for entry in entries
            if entry.role is not AdjacencyRole.STRUCTURAL
            and (role is None or entry.role is role)
        ]

    def strict_edge_filter(
        self,
        vertex_id: Any,
        predicate: Callable[[EdgeRecord], bool],
        role: Optional[AdjacencyRole] = None,
    ) -> list[EdgeRecord]:
        """Evaluate an edge predicate under a fixed, degree-padded schedule.

        This reference method records the public read counts.  The concrete
        implementation must issue dummy Vertex/Edge ORAM paths for missing or
        structural slots before filtering decrypted edge payloads locally.
        """
        self.apply_pending_updates(vertex_id)
        vertex = self.vertices[vertex_id]
        local_ids = {vertex_id}
        accessed_edge_ids = set()

        for entry in vertex.adjacency.values():
            self.apply_pending_updates(entry.neighbor_id)
            local_ids.add(entry.neighbor_id)
            if entry.edge_valid:
                accessed_edge_ids.add(entry.edge_id)

        matches = []
        for entry in vertex.adjacency.values():
            if not entry.edge_valid or (role is not None and entry.role is not role):
                continue
            edge = self.edges[entry.edge_id]
            if predicate(edge):
                matches.append(edge)

        self._remap_local(local_ids, accessed_edge_ids)
        real_insertions = self.last_schedule["meta_insertions"]
        self.last_schedule = {
            "vertex_reads": 1 + self.max_degree,
            "edge_reads": self.max_degree,
            "meta_insertions": real_insertions,
        }
        return matches

    def assert_reference_consistency(self, materialize_pending: bool = True) -> None:
        """Check direct references or their newest pending coverage.

        With ``materialize_pending=True``, the method consumes all pending
        blocks and checks exact equality. Otherwise it leaves protocol state
        untouched and accepts a stale field only when the matching newest
        pending field covers the current object path.
        """
        if materialize_pending:
            self.apply_all_pending_updates()

        for vertex in self.vertices.values():
            if len(vertex.adjacency) > self.max_degree:
                raise AssertionError("Vertex exceeds max_degree.")
            for neighbor_id, entry in vertex.adjacency.items():
                neighbor = self.vertices[neighbor_id]
                if entry.neighbor_path != neighbor.path:
                    pending = self._pending.get(vertex.vertex_id, {}).get(neighbor_id)
                    if not (
                        not materialize_pending
                        and pending is not None
                        and pending.vertex_valid
                        and pending.new_vertex_path == neighbor.path
                    ):
                        raise AssertionError("Uncovered stale neighbor path.")
                reverse = neighbor.adjacency.get(vertex.vertex_id)
                if reverse is None:
                    raise AssertionError("Missing reverse adjacency entry.")
                if entry.edge_valid:
                    edge = self.edges[entry.edge_id]
                    if entry.edge_path != edge.path:
                        pending = self._pending.get(vertex.vertex_id, {}).get(neighbor_id)
                        if not (
                            not materialize_pending
                            and pending is not None
                            and pending.edge_valid
                            and pending.edge_id == entry.edge_id
                            and pending.new_edge_path == edge.path
                        ):
                            raise AssertionError("Uncovered stale edge path.")
                    if reverse.edge_id != entry.edge_id:
                        raise AssertionError("Endpoint edge ids disagree.")
