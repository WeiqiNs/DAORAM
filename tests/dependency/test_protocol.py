import pickle

import msgpack
import pytest

from oblivlib.dependency import (
    DuplicateLabelError,
    ListPopBack,
    ListPushFront,
    ListWrite,
    ProtocolError,
    ServerError,
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
    DropTree,
    ErrorReply,
    Hello,
    HelloReply,
    ListOps,
    Ok,
    ReadList,
    ReadPath,
    Resize,
    Shutdown,
    WritePath,
    WriteRange,
    decode_reply,
    decode_request,
    encode,
    error_reply,
    expect_reply,
)


def test_every_message_round_trips():
    requests = [
        Hello(version=1),
        CreateTree(session="s", label="t", level=3, row_bytes=64),
        CreateTree(session="s", label="t", level=3, row_bytes=None),
        CreateList(session="s", label="l"),
        AttachTree(session="s", label="t", filename="f.tree", level=2, row_bytes=8),
        DropTree(session="s", label="t"),
        Resize(session="s", label="t", level=4),
        WriteRange(session="s", label="t", start=5, rows=[b"a", b""]),
        Batch(
            session="s",
            writes=[
                WritePath(label="t", leaves=[0, 3], rows=[b"x", b"", b"z"]),
                ListOps(label="l", ops=[ListWrite(0, b"v"), ListPushFront(b"w"), ListPopBack()]),
            ],
            reads=[
                ReadPath(label="t", leaves=[1]),
                ReadList(label="l", indices=None),
                ReadList(label="l", indices=[2]),
            ],
        ),
        Batch(session="s", writes=[], reads=[]),
        Close(session="s"),
        Shutdown(),
    ]
    replies = [
        HelloReply(session="s", max_message_bytes=1024),
        Ok(),
        BatchReply(results=[[b"r", b""], []]),
        ErrorReply(kind="RowSizeError", message="m"),
    ]
    for request in requests:
        assert decode_request(encode(request)) == request
    for reply in replies:
        assert decode_reply(encode(reply)) == reply


def test_decode_rejects_malformed_payloads():
    valid = msgpack.packb(["CreateList", "s", "l"])
    assert decode_request(valid) == CreateList(session="s", label="l")

    cases = [
        msgpack.packb(["NoSuchMessage", "s"]),
        msgpack.packb(["CreateList", "s"]),
        msgpack.packb(["CreateList", "s", "l", "extra"]),
        msgpack.packb(["Resize", "s", "t", True]),
        msgpack.packb(["WriteRange", "s", "t", 0, ["not bytes"]]),
        msgpack.packb(["Batch", "s", [["ReadPath", "t", [0]]], []]),
        msgpack.packb(["Batch", "s", [["ListOps", "l", [["ListWrite", -1, b"v"]]]], []]),
        msgpack.packb(["Ok"]),
        msgpack.packb({"tag": "CreateList"}),
        pickle.dumps(("execute", {})),
        valid[:-1],
    ]
    for payload in cases:
        with pytest.raises(ProtocolError):
            decode_request(payload)


def test_error_replies_map_to_error_classes():
    for exc, kind in [
        (UnknownLabelError("tree 't' is not hosted"), UnknownLabelError),
        (DuplicateLabelError("dup"), DuplicateLabelError),
        (FileNotFoundError("gone"), StorageError),
        (RuntimeError("boom"), ServerError),
    ]:
        reply = decode_reply(encode(error_reply(exc)))
        with pytest.raises(kind, match=str(exc)):
            expect_reply(reply, Ok)

    assert expect_reply(Ok(), Ok) == Ok()
    with pytest.raises(ProtocolError, match="unknown error kind"):
        expect_reply(ErrorReply(kind="KeyboardInterrupt", message="m"), Ok)
    with pytest.raises(ProtocolError, match="expected BatchReply"):
        expect_reply(Ok(), BatchReply)
