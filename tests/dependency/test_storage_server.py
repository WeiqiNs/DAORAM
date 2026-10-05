import pytest

from oblivlib.dependency import (
    DuplicateLabelError,
    ListPopBack,
    ListPushFront,
    ListWrite,
    ProtocolError,
    RowSizeError,
    ScaleDownError,
    StorageError,
    UnknownLabelError,
)
from oblivlib.dependency.protocol import (
    AttachTree,
    Batch,
    BatchReply,
    Close,
    CreateList,
    CreateTree,
    Hello,
    HelloReply,
    ListOps,
    Ok,
    ReadList,
    ReadPath,
    Reply,
    Resize,
    WritePath,
    WriteRange,
    expect_reply,
)
from oblivlib.dependency.storage_server import StorageServer

ROW = 4


@pytest.fixture(params=["memory", "disk"])
def server(request, tmp_path):
    storage = StorageServer(tmp_path / "store" if request.param == "disk" else None)
    yield storage
    storage.close()


def _open(server: StorageServer) -> str:
    return expect_reply(server.handle(Hello(version=1)), HelloReply).session


def _ok(reply: Reply) -> None:
    expect_reply(reply, Ok)


def _read(server: StorageServer, session: str, *reads: ReadPath | ReadList) -> list[list[bytes]]:
    return expect_reply(server.handle(Batch(session=session, writes=[], reads=list(reads))), BatchReply).results


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def test_hello_opens_sessions_and_checks_version(server):
    reply = expect_reply(server.handle(Hello(version=1)), HelloReply)
    assert len(reply.session) == 32 and reply.max_message_bytes == server.max_message_bytes

    with pytest.raises(ProtocolError, match="version 2"):
        expect_reply(server.handle(Hello(version=2)), HelloReply)
    with pytest.raises(ProtocolError, match="unknown session"):
        _ok(server.handle(CreateList(session="nope", label="l")))


def test_write_range_then_read_path_root_first_deduped(server):
    session = _open(server)
    _ok(server.handle(CreateTree(session=session, label="t", level=3, row_bytes=ROW)))
    _ok(server.handle(WriteRange(session=session, label="t", start=0, rows=[_row(1), _row(2), b"", _row(4)])))

    results = _read(server, session, ReadPath(label="t", leaves=[1, 0, 1]), ReadPath(label="t", leaves=[3]))
    assert results == [[_row(1), _row(2), _row(4), b""], [_row(1), b"", b""]]

    with pytest.raises(ProtocolError, match="outside"):
        _ok(server.handle(WriteRange(session=session, label="t", start=6, rows=[_row(1), _row(1)])))


def test_batch_is_validated_before_any_write(server):
    session = _open(server)
    for label in ("good", "t"):
        _ok(server.handle(CreateTree(session=session, label=label, level=2, row_bytes=ROW)))
    _ok(server.handle(CreateList(session=session, label="l")))
    good = WritePath(label="good", leaves=[0], rows=[_row(9), _row(9)])

    cases: list[tuple[list[WritePath | ListOps], list[ReadPath | ReadList], type[Exception]]] = [
        ([WritePath(label="missing", leaves=[0], rows=[_row(1), _row(1)])], [], UnknownLabelError),
        ([WritePath(label="t", leaves=[2], rows=[_row(1), _row(1)])], [], ProtocolError),
        ([WritePath(label="t", leaves=[0], rows=[_row(1)])], [], ProtocolError),
        ([WritePath(label="t", leaves=[0], rows=[_row(1), b"short"])], [], RowSizeError),
        ([ListOps(label="l", ops=[ListPushFront(b"a"), ListWrite(1, b"b")])], [], ProtocolError),
        ([ListOps(label="l", ops=[ListPopBack()])], [], ProtocolError),
        ([ListOps(label="l", ops=[ListPushFront(b"a")])], [ReadList(label="l", indices=[1])], ProtocolError),
        ([], [ReadPath(label="t", leaves=[-1])], ProtocolError),
    ]
    for writes, reads, error in cases:
        with pytest.raises(error):
            expect_reply(server.handle(Batch(session=session, writes=[good, *writes], reads=reads)), BatchReply)
        assert _read(server, session, ReadPath(label="good", leaves=[0]), ReadList(label="l", indices=None)) == [
            [b"", b""],
            [],
        ]


def test_list_ops_apply_in_order_before_reads(server):
    session = _open(server)
    _ok(server.handle(CreateList(session=session, label="l")))
    batch = Batch(
        session=session,
        writes=[
            ListOps(label="l", ops=[ListPushFront(b"a"), ListPushFront(b"b"), ListPushFront(b"c")]),
            ListOps(label="l", ops=[ListPopBack(), ListWrite(0, b"z")]),
        ],
        reads=[ReadList(label="l", indices=None), ReadList(label="l", indices=[1])],
    )
    assert expect_reply(server.handle(batch), BatchReply).results == [[b"z", b"b"], [b"b"]]


def test_variable_rows_are_memory_only(tmp_path):
    memory = StorageServer()
    session = _open(memory)
    _ok(memory.handle(CreateTree(session=session, label="t", level=2, row_bytes=None)))
    write = WritePath(label="t", leaves=[1], rows=[b"root bucket", b"x"])
    expect_reply(memory.handle(Batch(session=session, writes=[write], reads=[])), BatchReply)
    assert _read(memory, session, ReadPath(label="t", leaves=[0, 1])) == [[b"root bucket", b"", b"x"]]

    disk = StorageServer(tmp_path)
    with pytest.raises(ProtocolError, match="fixed-size"):
        _ok(disk.handle(CreateTree(session=_open(disk), label="t", level=2, row_bytes=None)))
    disk.close()


def test_resize_grows_absent_and_refuses_dropping_present(server):
    session = _open(server)
    _ok(server.handle(CreateTree(session=session, label="t", level=2, row_bytes=ROW)))
    _ok(server.handle(WriteRange(session=session, label="t", start=0, rows=[_row(1), _row(2), b""])))

    _ok(server.handle(Resize(session=session, label="t", level=3)))
    assert _read(server, session, ReadPath(label="t", leaves=[0, 3])) == [[_row(1), _row(2), b"", b"", b""]]

    _ok(server.handle(Resize(session=session, label="t", level=2)))
    with pytest.raises(ScaleDownError):
        _ok(server.handle(Resize(session=session, label="t", level=1)))
    with pytest.raises(ScaleDownError):
        _ok(server.handle(Resize(session=session, label="t", level=0)))
    assert _read(server, session, ReadPath(label="t", leaves=[0])) == [[_row(1), _row(2)]]


def test_attach_adopts_file_and_rejects_unsafe_names(tmp_path):
    store_dir = tmp_path / "store"
    server = StorageServer(store_dir)
    session = _open(server)
    image = store_dir / "image.tree"
    image.write_bytes(_row(1) + _row(2) + bytes(ROW))

    with pytest.raises(RowSizeError):
        _ok(server.handle(AttachTree(session=session, label="t", filename="image.tree", level=3, row_bytes=ROW)))
    for name in ("../image.tree", str(image), "a/image.tree", "..", "", session, "missing.tree"):
        with pytest.raises((ProtocolError, StorageError)):
            _ok(server.handle(AttachTree(session=session, label="t", filename=name, level=2, row_bytes=ROW)))

    _ok(server.handle(AttachTree(session=session, label="t", filename="image.tree", level=2, row_bytes=ROW)))
    assert not image.exists()
    assert _read(server, session, ReadPath(label="t", leaves=[0, 1])) == [[_row(1), _row(2), b""]]

    memory = StorageServer()
    with pytest.raises(ProtocolError, match="in-memory"):
        _ok(memory.handle(AttachTree(session=_open(memory), label="t", filename="x.tree", level=2, row_bytes=ROW)))
    server.close()


def test_close_deletes_session_files_and_labels_are_per_session(tmp_path):
    store_dir = tmp_path / "store"
    server = StorageServer(store_dir)
    first, second = _open(server), _open(server)
    for session in (first, second):
        _ok(server.handle(CreateTree(session=session, label="t", level=2, row_bytes=ROW)))
    with pytest.raises(DuplicateLabelError):
        _ok(server.handle(CreateList(session=first, label="t")))
    assert (store_dir / first).is_dir()

    _ok(server.handle(Close(session=first)))
    assert not (store_dir / first).exists() and (store_dir / second).is_dir()
    with pytest.raises(ProtocolError, match="unknown session"):
        _read(server, first, ReadPath(label="t", leaves=[0]))

    server.close()
    assert not (store_dir / second).exists()
