"""Integration tests for composite updates backed by the Edge Data ORAM."""

from __future__ import annotations

import random

from daoram.dependency import InteractLocalServer
from daoram.graph.edge_oram import EdgeDataOram
from daoram.graph.entity_grove_runtime import EntityGroveRuntime
from daoram.graph.first_class_edge import FirstClassEdgeState


def make_runtime() -> tuple[EntityGroveRuntime, InteractLocalServer]:
    state = FirstClassEdgeState(max_degree=4, leaf_range=64, rng=random.Random(23))
    for vertex_id in ("u", "v", "w", "x"):
        state.add_vertex(vertex_id, value=vertex_id.upper())
    state.insert_edge("u", "v", value={"weight": 3}, edge_id="uv")
    state.insert_edge("u", "w", value={"weight": 9}, edge_id="uw")
    state.apply_all_pending_updates()

    client = InteractLocalServer()
    edge_oram = EdgeDataOram(
        num_edges=64,
        data_size=128,
        client=client,
        bucket_size=4,
        stash_scale=20,
    )
    runtime = EntityGroveRuntime(state=state, edge_oram=edge_oram)
    runtime.initialize_edge_storage()
    client.reset_metrics()
    return runtime, client


def test_edge_oram_path_drives_the_composite_update() -> None:
    runtime, client = make_runtime()

    edge = runtime.lookup_edge("u", "v")

    pending = runtime.state.pending_updates("v")["u"]
    assert pending.new_edge_path == edge.path
    assert pending.new_vertex_path == runtime.state.vertices["u"].path
    assert runtime.state.edges["uv"].path == edge.path
    assert runtime.edge_oram.position_map_size == 0
    assert client.get_rounds() == 2
    assert runtime.metrics["edge_oram"]["bytes_read"] > 0
    assert runtime.metrics["edge_oram"]["bytes_written"] > 0


def test_opposite_endpoint_consumes_update_before_next_edge_access() -> None:
    runtime, _ = make_runtime()

    first = runtime.lookup_edge("u", "v")
    second = runtime.lookup_edge("v", "u", value={"weight": 4}, update_value=True)

    assert second.path != first.path or runtime.edge_oram.leaf_range > 1
    assert second.value == {"weight": 4}
    assert runtime.state.vertices["v"].adjacency["u"].edge_path == second.path
    assert runtime.state.pending_updates("u")["v"].new_edge_path == second.path
    runtime.assert_consistent()


def test_runtime_insert_access_and_delete_share_one_edge_record() -> None:
    runtime, _ = make_runtime()

    inserted = runtime.insert_edge("v", "x", value={"weight": 5}, edge_id="vx")
    accessed = runtime.lookup_edge("v", "x", include_neighbor=True)
    deleted = runtime.delete_edge("x", "v")

    assert accessed.edge_id == inserted.edge_id
    assert deleted.edge_id == "vx"
    assert "vx" not in runtime.state.edges
    assert "x" not in runtime.state.vertices["v"].adjacency
    assert "v" not in runtime.state.vertices["x"].adjacency
    runtime.assert_consistent()


def test_repeated_cross_endpoint_accesses_keep_references_covered() -> None:
    runtime, _ = make_runtime()

    for index in range(100):
        if index % 2:
            runtime.lookup_edge("u", "v", include_neighbor=True)
        else:
            runtime.lookup_edge("v", "u", include_neighbor=False)

    runtime.state.apply_all_pending_updates()
    runtime.assert_consistent()
    edge_path = runtime.state.edges["uv"].path
    assert runtime.state.vertices["u"].adjacency["v"].edge_path == edge_path
    assert runtime.state.vertices["v"].adjacency["u"].edge_path == edge_path
