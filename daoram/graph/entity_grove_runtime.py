"""Runtime bridge between entity references and the concrete Edge Data ORAM."""

from __future__ import annotations

from typing import Any, Optional

from daoram.graph.edge_oram import EdgeDataOram, EdgeReference
from daoram.graph.first_class_edge import EdgeRecord, FirstClassEdgeState


class EntityGroveRuntime:
    """Keep the executable entity model and Edge Data ORAM on one path state.

    Vertex Data and Vertex Meta accesses are still represented by
    :class:`FirstClassEdgeState`. Edge reads, remaps, insertions, and deletions
    execute against the concrete ORAM. This bridge makes the ORAM-selected edge
    path authoritative for both endpoint references and pending updates.
    """

    def __init__(self, state: FirstClassEdgeState, edge_oram: EdgeDataOram) -> None:
        if state.leaf_range != edge_oram.leaf_range:
            raise ValueError("State and Edge Data ORAM must use the same leaf range.")
        self.state = state
        self.edge_oram = edge_oram

    def initialize_edge_storage(self) -> None:
        """Initialize concrete edge storage from the current logical records."""
        self.edge_oram.init_server_storage(self.state.edges.values())
        self.assert_consistent()

    def lookup_edge(
        self,
        endpoint_id: Any,
        neighbor_id: Any,
        include_neighbor: bool = False,
        value: Optional[Any] = None,
        update_value: bool = False,
    ) -> EdgeRecord:
        """Access and remap an edge using the endpoint-owned current path."""
        self.state.apply_pending_updates(endpoint_id)
        logical_edge = self.state._require_edge(endpoint_id, neighbor_id)
        entry = self.state.vertices[endpoint_id].adjacency[neighbor_id]
        value_updates = {logical_edge.edge_id: value} if update_value else None
        concrete_edge = self.edge_oram.access(
            [EdgeReference(logical_edge.edge_id, entry.edge_path)],
            value_updates=value_updates,
        )[logical_edge.edge_id]

        remapped = self.state.lookup_edge(
            endpoint_id=endpoint_id,
            neighbor_id=neighbor_id,
            include_neighbor=include_neighbor,
            new_edge_path=concrete_edge.path,
        )
        remapped.value = concrete_edge.value
        self.assert_consistent()
        return remapped

    def insert_edge(
        self,
        source_id: Any,
        target_id: Any,
        value: Any = None,
        edge_id: Optional[Any] = None,
        directed: bool = True,
    ) -> EdgeRecord:
        """Insert one logical edge and its matching concrete ORAM record."""
        record = self.state.insert_edge(
            source_id=source_id,
            target_id=target_id,
            value=value,
            edge_id=edge_id,
            directed=directed,
        )
        self.edge_oram.insert(record)
        self.assert_consistent()
        return record

    def delete_edge(self, endpoint_id: Any, neighbor_id: Any) -> EdgeRecord:
        """Delete one concrete edge and both endpoint-owned references."""
        self.state.apply_pending_updates(endpoint_id)
        logical_edge = self.state._require_edge(endpoint_id, neighbor_id)
        entry = self.state.vertices[endpoint_id].adjacency[neighbor_id]
        concrete_edge = self.edge_oram.delete(
            EdgeReference(logical_edge.edge_id, entry.edge_path)
        )
        deleted = self.state.delete_edge(endpoint_id, neighbor_id)
        if concrete_edge.edge_id != deleted.edge_id:
            raise AssertionError("Logical and concrete edge deletion diverged.")
        self.assert_consistent()
        return deleted

    def assert_consistent(self) -> None:
        """Check reference coverage and the endpoint-owned position-map rule."""
        self.state.assert_reference_consistency(materialize_pending=False)
        if self.edge_oram.position_map_size != 0:
            raise AssertionError("Edge Data ORAM unexpectedly retained a position map.")

    @property
    def metrics(self) -> dict[str, Any]:
        """Return concrete Edge Data ORAM metrics and logical schedule counts."""
        return {
            "edge_oram": self.edge_oram.metrics,
            "logical_schedule": dict(self.state.last_schedule),
        }
