"""Behavioral tests for the first-class-edge reference state machine."""

import random

import pytest

from daoram.graph.first_class_edge import AdjacencyRole, FirstClassEdgeState


def make_triangle() -> FirstClassEdgeState:
    state = FirstClassEdgeState(max_degree=4, leaf_range=64, rng=random.Random(7))
    for vertex_id in ("u", "v", "w"):
        state.add_vertex(vertex_id, value=vertex_id.upper())
    state.insert_edge("u", "v", value={"weight": 3}, edge_id="uv")
    state.insert_edge("u", "w", value={"weight": 9}, edge_id="uw")
    state.apply_all_pending_updates()
    state.assert_reference_consistency()
    return state


def test_edge_only_access_combines_vertex_and_edge_updates() -> None:
    state = make_triangle()

    edge = state.lookup_edge("u", "v", include_neighbor=False)

    update_for_v = state.pending_updates("v")["u"]
    assert update_for_v.vertex_valid
    assert update_for_v.edge_valid
    assert update_for_v.new_vertex_path == state.vertices["u"].path
    assert update_for_v.new_edge_path == edge.path

    update_for_w = state.pending_updates("w")["u"]
    assert update_for_w.vertex_valid
    assert not update_for_w.edge_valid
    assert state.last_schedule["meta_insertions"] == 2

    state.assert_reference_consistency()


def test_edge_and_neighbor_access_repairs_local_pair_directly() -> None:
    state = make_triangle()

    edge = state.lookup_edge("u", "v", include_neighbor=True)

    assert "u" not in state.pending_updates("v")
    assert "v" not in state.pending_updates("u")
    assert state.vertices["u"].adjacency["v"].edge_path == edge.path
    assert state.vertices["v"].adjacency["u"].edge_path == edge.path
    assert state.last_schedule["meta_insertions"] == 1
    state.assert_reference_consistency()


def test_pending_blocks_deduplicate_by_adjacency_entry() -> None:
    state = make_triangle()

    state.lookup_vertex("u")
    first = state.pending_updates("v")["u"]
    state.lookup_edge("u", "v", include_neighbor=False)
    merged = state.pending_updates("v")["u"]

    assert len(state.pending_updates("v")) == 1
    assert merged.epoch > first.epoch
    assert merged.vertex_valid and merged.edge_valid
    assert merged.new_vertex_path == state.vertices["u"].path
    assert merged.new_edge_path == state.edges["uv"].path
    state.assert_reference_consistency()


def test_insert_and_delete_edge_clean_both_endpoints() -> None:
    state = FirstClassEdgeState(max_degree=3, leaf_range=32, rng=random.Random(11))
    state.add_vertex("u")
    state.add_vertex("v")

    state.insert_edge("u", "v", edge_id="uv")
    assert state.vertices["u"].adjacency["v"].edge_id == "uv"
    assert state.vertices["v"].adjacency["u"].edge_id == "uv"
    assert not state.pending_updates("u")
    assert not state.pending_updates("v")

    state.delete_edge("u", "v")
    assert "uv" not in state.edges
    assert "v" not in state.vertices["u"].adjacency
    assert "u" not in state.vertices["v"].adjacency
    state.assert_reference_consistency()


def test_delete_vertex_immediately_removes_incident_edges() -> None:
    state = make_triangle()

    state.delete_vertex("u")

    assert "u" not in state.vertices
    assert not state.edges
    assert "u" not in state.vertices["v"].adjacency
    assert "u" not in state.vertices["w"].adjacency
    state.assert_reference_consistency()


def test_simple_directed_graph_uses_local_role_bits() -> None:
    state = FirstClassEdgeState(max_degree=3, leaf_range=32, rng=random.Random(13))
    state.add_vertex("u")
    state.add_vertex("v")
    state.insert_edge("u", "v", edge_id="uv", directed=True)

    assert state.neighbors("u", role=AdjacencyRole.OUTGOING) == ["v"]
    assert state.neighbors("u", role=AdjacencyRole.INCOMING) == []
    assert state.neighbors("v", role=AdjacencyRole.INCOMING) == ["u"]
    with pytest.raises(ValueError, match="Parallel edges"):
        state.insert_edge("v", "u", edge_id="vu", directed=True)


def test_structural_link_has_no_edge_record() -> None:
    state = FirstClassEdgeState(max_degree=3, leaf_range=32, rng=random.Random(17))
    state.add_vertex("u")
    state.add_vertex("i")

    state.insert_structural_link("u", "i")

    assert not state.edges
    assert not state.vertices["u"].adjacency["i"].edge_valid
    assert state.vertices["u"].adjacency["i"].role is AdjacencyRole.STRUCTURAL
    assert state.neighbors("u") == []
    state.assert_reference_consistency()


def test_strict_filter_has_degree_padded_public_schedule() -> None:
    state = make_triangle()

    matches = state.strict_edge_filter(
        "u", predicate=lambda edge: edge.value["weight"] >= 5
    )

    assert [edge.edge_id for edge in matches] == ["uw"]
    assert state.last_schedule["vertex_reads"] == 1 + state.max_degree
    assert state.last_schedule["edge_reads"] == state.max_degree
    state.assert_reference_consistency()


def test_delete_edge_schedule_includes_the_edge_record() -> None:
    state = make_triangle()

    state.delete_edge("u", "v")

    assert state.last_schedule["vertex_reads"] == 2
    assert state.last_schedule["edge_reads"] == 1


def test_delete_vertex_uses_the_degree_padded_cleanup_schedule() -> None:
    state = make_triangle()

    state.delete_vertex("u")

    assert state.last_schedule["vertex_reads"] == 1 + state.max_degree
    assert state.last_schedule["edge_reads"] == state.max_degree


@pytest.mark.parametrize("seed", range(10))
def test_random_operation_sequences_preserve_all_references(seed: int) -> None:
    rng = random.Random(seed)
    state = FirstClassEdgeState(max_degree=5, leaf_range=128, rng=rng)
    vertex_ids = list(range(10))
    for vertex_id in vertex_ids:
        state.add_vertex(vertex_id)

    for step in range(200):
        live_pairs = [tuple(edge_key) for edge_key in state._edge_by_pair]
        operation = rng.choice(("vertex", "edge", "insert", "delete"))

        if operation == "vertex":
            state.lookup_vertex(rng.choice(vertex_ids))
        elif operation == "edge" and live_pairs:
            left, right = rng.choice(live_pairs)
            state.lookup_edge(left, right, include_neighbor=bool(rng.getrandbits(1)))
        elif operation == "delete" and live_pairs:
            left, right = rng.choice(live_pairs)
            state.delete_edge(left, right)
        elif operation == "insert":
            candidates = [
                (left, right)
                for left in vertex_ids
                for right in vertex_ids
                if left < right
                and frozenset((left, right)) not in state._edge_by_pair
                and len(state.vertices[left].adjacency) < state.max_degree
                and len(state.vertices[right].adjacency) < state.max_degree
            ]
            if candidates:
                left, right = rng.choice(candidates)
                state.insert_edge(
                    left,
                    right,
                    edge_id=f"e-{seed}-{step}",
                    directed=bool(rng.getrandbits(1)),
                )

        state.assert_reference_consistency()
