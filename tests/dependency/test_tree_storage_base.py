import pytest

from oblivlib.dependency import (
    AesGcm,
    AvlOmapConfig,
    Data,
    KVPair,
    MissingClientError,
    PathOramConfig,
    StashOverflowError,
)
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.omap import AVLOmap


def test_missing_client_raises_identified_error():
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, name="orphan"))
    with pytest.raises(MissingClientError, match="TreeStorageBase 'orphan' has no client"):
        _ = base._client


def test_stash_overflow_raises():
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, stash_scale=1, name="tiny"))
    base._stash = [Data(key=i, leaf=0) for i in range(base._stash_capacity)]
    base._check_stash()

    base._stash.append(Data(key=99, leaf=0))
    with pytest.raises(StashOverflowError, match="TreeStorageBase 'tiny'") as raised:
        base._check_stash()
    assert isinstance(raised.value, MemoryError)


def _path_oram_base(encryptor):
    base = TreeStorageBase(PathOramConfig(num_data=8, data_size=8, encryptor=encryptor))
    return base, [Data(key=i, leaf=i % 2, value=bytes([i]) * 8) for i in range(8)]


def _avl_omap(encryptor):
    omap = AVLOmap(AvlOmapConfig(num_data=8, key_size=4, data_size=8, encryptor=encryptor))
    blocks, _ = omap._build_ods_blocks([KVPair(key=i, value=bytes([i]) * 8) for i in range(8)])
    return omap, blocks


@pytest.mark.parametrize("make_scheme", [_path_oram_base, _avl_omap], ids=["default_codec", "node_codec"])
def test_init_seal_matches_per_op_bucket_layout(make_scheme):
    encryptor = AesGcm()
    scheme, blocks = make_scheme(encryptor)
    tree = scheme._build_tree(blocks)

    occupied = 0
    for index in range(tree.size):
        init_blob = tree.storage[index][0]
        assert isinstance(init_blob, bytes)
        bucket = scheme._decrypt_path_data({index: [init_blob]})[index]
        occupied += len(bucket)
        per_op_blob = scheme._encrypt_path_data({index: list(bucket)})[index][0]
        assert isinstance(per_op_blob, bytes)
        assert encryptor.dec(per_op_blob) == encryptor.dec(init_blob)
    assert occupied + len(scheme._stash) == len(blocks)


def test_plaintext_decrypt_drops_dummies():
    blocks = [Data(), Data(key=1, leaf=0, value=b"a"), Data()]
    plain = TreeStorageBase(PathOramConfig(num_data=4, data_size=4))
    sealed = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, encryptor=AesGcm()))

    expected = {0: [Data(key=1, leaf=0, value=b"a")]}
    assert plain._decrypt_path_data({0: list(blocks)}) == expected
    assert sealed._decrypt_path_data(sealed._encrypt_path_data({0: list(blocks)})) == expected
