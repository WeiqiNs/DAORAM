import pytest

from oblivlib.dependency import (
    Client,
    ListPopBack,
    ListPushFront,
    ListWrite,
    LocalBackend,
    Metrics,
    ProtocolError,
    RowSizeError,
    StorageServer,
    UnknownLabelError,
)
from oblivlib.dependency.protocol import WriteRange
from oblivlib.dependency.tree_builder import MemoryImage

ROW = 4


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def _tree(client: Client, label: str = "t", level: int = 3) -> None:
    client.host_tree(
        label, MemoryImage(level=level, row_bytes=ROW, rows=[_row(i + 1) for i in range((1 << level) - 1)])
    )


def test_local_client_hosts_rows_and_reads_root_first():
    with Client.local() as client:
        _tree(client)
        client.add_read_path("t", [3, 0])
        rows = client.execute().require("t")
        assert list(rows) == [0, 1, 2, 3, 6]
        assert list(rows.values()) == [_row(1), _row(2), _row(3), _row(4), _row(7)]


def test_metrics_count_rounds_and_payload_and_reset():
    with Client.local(defer_writes=False) as client:
        _tree(client)
        client.reset_metrics()
        client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
        client.execute()
        client.add_read_path("t", [0])
        client.execute()
        assert client.metrics == Metrics(rounds=2, payload_bytes=6 * ROW, wire_bytes=0)

        client.reset_metrics()
        assert client.metrics == Metrics(rounds=0, payload_bytes=0, wire_bytes=0)


def test_staging_checks_known_labels_and_whole_paths():
    with Client.local() as client:
        _tree(client)
        with pytest.raises(UnknownLabelError, match="'missing'"):
            client.add_read_path("missing", [0])
        with pytest.raises(UnknownLabelError, match="'t'"):
            client.add_read_list("t", None)
        for partial in ({0: _row(1), 1: _row(1)}, {0: _row(1), 1: _row(1), 3: _row(1), 4: _row(1), 2: _row(1)}):
            with pytest.raises(ValueError, match="'t'"):
                client.add_write_path("t", partial)
        client.add_read_path("t", [0])
        assert client.execute().require("t")[3] == _row(4)


def test_writes_apply_before_reads_and_later_write_wins():
    with Client.local(defer_writes=False) as client:
        _tree(client)
        client.add_write_path("t", {0: _row(10), 1: _row(10), 3: _row(10)})
        client.add_write_path("t", {0: _row(20), 1: _row(20), 4: _row(20)})
        client.add_read_path("t", [0, 1])
        rows = client.execute().require("t")
        assert rows == {0: _row(20), 1: _row(20), 3: _row(10), 4: _row(20)}


def test_list_ops_and_reads():
    with Client.local() as client:
        client.create_list("l")
        client.add_write_list("l", [ListPushFront(b"a"), ListPushFront(b"b")])
        client.add_write_list("l", [ListPushFront(b"c"), ListPushFront(b"d"), ListPopBack(), ListWrite(1, b"z")])
        client.add_read_list("l", [2])
        client.add_read_list("l", [0])
        assert client.execute().require("l") == {0: b"d", 2: b"b"}

        client.add_read_list("l", [1])
        client.add_read_list("l", None)
        assert client.execute().require("l") == [b"d", b"z", b"b"]


def test_deferred_writes_ride_next_execute(recorded_client):
    for defer_writes, rounds_after_write in [(True, 0), (False, 1)]:
        client, recorder = recorded_client(defer_writes=defer_writes)
        _tree(client)
        client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
        assert client.execute().results == {}
        assert client.metrics.rounds == rounds_after_write

        client.add_read_path("t", [0])
        assert client.execute().require("t")[3] == _row(9)
        assert client.metrics.rounds == rounds_after_write + 1
        assert bool(recorder.batches[-1].writes) is defer_writes


def test_empty_execute_is_a_round():
    with Client.local() as client:
        assert client.execute().results == {}
        assert client.metrics.rounds == 1


def test_deferred_write_error_surfaces_on_next_call():
    with Client.local() as client:
        _tree(client, level=2)
        client.add_write_path("t", {0: b"short", 1: _row(9)})
        assert client.execute().results == {}

        client.add_read_path("t", [0])
        with pytest.raises(RowSizeError):
            client.execute()

        client.add_read_path("t", [0])
        assert client.execute().require("t") == {0: _row(1), 1: _row(2)}


def test_host_tree_streams_byte_bounded_chunks(recorded_client):
    rows = [bytes([i + 1]) * 100 for i in range(100)]
    client, recorder = recorded_client(server=StorageServer(max_message_bytes=4096))
    client.host_tree("t", MemoryImage(level=7, row_bytes=100, rows=rows))

    chunks = [message for message in recorder.messages if isinstance(message, WriteRange)]
    assert [(chunk.start, len(chunk.rows)) for chunk in chunks] == [(0, 20), (20, 20), (40, 20), (60, 20), (80, 20)]
    client.add_read_path("t", range(64))
    stored = client.execute().require("t")
    assert [stored[i] for i in range(100)] == rows
    assert stored[100] == b""


def test_sessions_namespace_labels():
    server = StorageServer()
    first, second = Client(LocalBackend(server)), Client(LocalBackend(server))
    first.host_tree("t", MemoryImage(level=1, row_bytes=ROW, rows=[_row(1)]))
    second.host_tree("t", MemoryImage(level=1, row_bytes=ROW, rows=[_row(2)]))
    for client, expected in [(first, _row(1)), (second, _row(2))]:
        client.add_read_path("t", [0])
        assert client.execute().require("t") == {0: expected}


def test_flush_and_close(recorded_client):
    client, recorder = recorded_client()
    with client:
        _tree(client)
        client.flush()
        assert client.metrics.rounds == 0

        client.add_write_path("t", {0: _row(9), 1: _row(9), 3: _row(9)})
        client.execute()
        client.flush()
        assert client.metrics.rounds == 1

        client.add_write_path("t", {0: _row(8), 1: _row(8), 4: _row(8)})
        client.execute()

    assert recorder.batches[-1].writes and client.metrics.rounds == 2
    client.close()
    with pytest.raises(ProtocolError, match="unknown session"):
        client.execute()
