from pathlib import Path

import msgpack
import pytest

import oblivlib
from oblivlib.dependency import AVLData, BPlusData, Data
from oblivlib.dependency.codec import DefaultCodec, NodeCodec


def _through_msgpack(fields: list) -> list:
    return msgpack.unpackb(msgpack.packb(fields))


def test_default_codec_round_trips_through_msgpack():
    block = Data(key=3, leaf=7, value=b"v")
    codec = DefaultCodec(max_block_bytes=64)

    assert codec.pack(block) == [3, 7, b"v"]
    assert codec.unpack(_through_msgpack(codec.pack(block))) == block


_NODE_CASES = [
    pytest.param(
        AVLData,
        AVLData(value=b"v", r_key=b"rk", r_leaf=3, r_height=2, l_key=b"lk", l_leaf=1, l_height=2),
        AVLData(value=b"v", r_key=b"rk", r_leaf=3, r_height=2, l_key=b"lk", l_leaf=1, l_height=2),
        id="avl",
    ),
    pytest.param(
        BPlusData,
        BPlusData(keys=[b"a", b"b"], values=[(10, 0), (20, 1), (30, 2)]),
        BPlusData(keys=[b"a", b"b"], values=[[10, 0], [20, 1], [30, 2]]),
        id="bplus_internal",
    ),
    pytest.param(
        BPlusData, BPlusData(keys=[b"a"], values=[b"x"]), BPlusData(keys=[b"a"], values=[b"x"]), id="bplus_leaf"
    ),
]


@pytest.mark.parametrize(("value_cls", "value", "expected"), _NODE_CASES)
def test_node_codec_round_trips_without_mutating(value_cls, value, expected):
    block = Data(key=b"k", leaf=4, value=value)
    codec = NodeCodec(max_block_bytes=256, value_cls=value_cls)

    loaded = codec.unpack(_through_msgpack(codec.pack(block)))

    assert block.value is value
    assert loaded == Data(key=b"k", leaf=4, value=expected)


def test_library_has_no_pickle():
    sources = Path(oblivlib.__file__).parent.rglob("*.py")
    assert [path.name for path in sources if "pickle" in path.read_text()] == []
