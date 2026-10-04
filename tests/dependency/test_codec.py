"""BlockCodec contract for the dependency-layer codecs.

``DefaultCodec`` stores a block as its own padded pickle; ``NodeCodec`` packs an ODS node whose
``value`` is a dataclass (``AVLData``/``BPlusData``) stored as its own pickle bytes, fixed-width. The
key invariant both pin: ``dump_block`` builds the payload from a copy and never mutates the live block.
"""

import pytest

from oblivlib.dependency import AesGcm, AVLData, BPlusData, Data
from oblivlib.dependency.codec import DefaultCodec, NodeCodec


def test_default_codec_round_trips_to_fixed_width():
    block = Data(key="k", leaf=7, value=[1, 2, 3])
    codec = DefaultCodec(block_size=256)

    payload = codec.dump_block(block)
    assert len(payload) == codec.block_size == 256
    assert block == Data(key="k", leaf=7, value=[1, 2, 3])

    loaded = codec.load_block(payload)
    assert loaded == block
    assert loaded.value == [1, 2, 3]


def test_default_codec_loads_dummy_as_dummy():
    codec = DefaultCodec(block_size=128)
    dummy = codec.load_block(codec.dummy_block())
    assert not dummy.is_real()
    assert dummy.key is None and dummy.value is None


def test_seal_open_bucket_round_trip():
    encryptor = AesGcm()
    codec = DefaultCodec(block_size=128)
    blocks = [Data(key=1, leaf=0, value="a"), Data(key=2, leaf=1, value="b")]

    blob = codec.seal_bucket(encryptor, blocks, bucket_size=3)
    assert len(blob) == encryptor.ciphertext_length(3 * codec.block_size)
    assert codec.open_bucket(encryptor, blob) == blocks
    with pytest.raises(ValueError):
        codec.seal_bucket(encryptor, blocks, bucket_size=1)


_NODE_CASES = [
    pytest.param(
        AVLData,
        AVLData(value=b"v", r_key=b"rk", r_leaf=3, r_height=2, l_key=b"lk", l_leaf=1, l_height=2),
        b"k",
        7,
        id="avl",
    ),
    pytest.param(BPlusData, BPlusData(keys=[1, 2], values=[(10, 0), (20, 1)]), 5, 4, id="bplus"),
]


@pytest.mark.parametrize(("value_cls", "value", "key", "leaf"), _NODE_CASES)
def test_node_codec_does_not_mutate_and_round_trips(value_cls, value, key, leaf):
    block = Data(key=key, leaf=leaf, value=value)
    codec = NodeCodec(block_size=512, value_cls=value_cls)

    payload = codec.dump_block(block)
    assert block.value is value
    assert isinstance(block.value, value_cls)
    assert len(payload) == 512

    loaded = codec.load_block(payload)
    assert loaded.key == key and loaded.leaf == leaf
    assert isinstance(loaded.value, value_cls)
    assert loaded.value.dump() == value.dump()


@pytest.mark.parametrize("value_cls", [AVLData, BPlusData], ids=["avl", "bplus"])
def test_node_codec_loads_dummy_as_dummy(value_cls):
    codec = NodeCodec(block_size=256, value_cls=value_cls)
    dummy = codec.load_block(codec.dummy_block())
    assert not dummy.is_real()
    assert dummy.value is None
