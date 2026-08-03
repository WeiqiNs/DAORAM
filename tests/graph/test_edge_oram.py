"""Tests for endpoint-owned access to the independent Edge Data ORAM."""

from __future__ import annotations

from daoram.dependency import InteractLocalServer
from daoram.graph.edge_oram import EdgeDataOram, EdgeReference
from daoram.graph.first_class_edge import EdgeRecord


def make_edge_oram() -> tuple[EdgeDataOram, dict[str, EdgeRecord]]:
    client = InteractLocalServer()
    edge_oram = EdgeDataOram(
        num_edges=32,
        data_size=64,
        client=client,
        bucket_size=4,
        stash_scale=20,
    )
    records = {
        "uv": EdgeRecord("uv", "u", "v", {"weight": 3}, path=2),
        "uw": EdgeRecord("uw", "u", "w", {"weight": 9}, path=7),
    }
    edge_oram.init_server_storage(records.values())
    return edge_oram, records


def test_initialization_discards_global_edge_position_map() -> None:
    edge_oram, _ = make_edge_oram()
    assert edge_oram.position_map_size == 0


def test_access_uses_endpoint_path_and_returns_new_reference() -> None:
    edge_oram, records = make_edge_oram()

    result = edge_oram.access(
        [EdgeReference("uv", records["uv"].path)],
        new_paths={"uv": 11},
    )

    assert result["uv"].value == {"weight": 3}
    assert result["uv"].path == 11
    assert edge_oram.position_map_size == 0

    again = edge_oram.access([EdgeReference("uv", 11)], new_paths={"uv": 13})
    assert again["uv"].path == 13


def test_batch_access_remaps_independent_edges_in_one_round() -> None:
    edge_oram, records = make_edge_oram()

    result = edge_oram.access(
        [
            EdgeReference("uv", records["uv"].path),
            EdgeReference("uw", records["uw"].path),
        ],
        value_updates={"uw": {"weight": 10}},
        new_paths={"uv": 12, "uw": 14},
    )

    assert result["uv"].path == 12
    assert result["uw"].path == 14
    assert result["uw"].value == {"weight": 10}


def test_insert_and_delete_need_no_global_edge_lookup() -> None:
    edge_oram, _ = make_edge_oram()
    record = EdgeRecord("vx", "v", "x", {"weight": 1}, path=5)

    edge_oram.insert(record, eviction_path=3)
    read = edge_oram.access([EdgeReference("vx", 5)], new_paths={"vx": 9})
    assert read["vx"].path == 9

    deleted = edge_oram.delete(EdgeReference("vx", 9))
    assert deleted.edge_id == "vx"
