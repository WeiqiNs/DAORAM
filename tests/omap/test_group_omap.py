"""Tests for GroupOmap (group-by-hash OMAP: upper ORAM for metadata + MulPathOram lower ORAM)."""

import random

import pytest

from oblivlib.dependency import AesGcm, DaOramConfig, GroupOmapConfig, PathOramConfig
from oblivlib.omap import GroupOmap
from oblivlib.oram import DAOram, PathOram


def _make_upper(kind, n, client, encryptor, key_size=10):
    # The upper ORAM must be wide enough for a full bucket's pickled metadata; size it via the helper.
    data_size = GroupOmap.upper_oram_data_size(num_data=n, key_size=key_size)
    if kind == "path":
        return PathOram(
            PathOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor)
        )
    return DAOram(DaOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor))


def _duplicate_blocks(omap):
    """Return {lower_key: count} for any key with more than one block (plaintext lower ORAM only).

    The group design relies on exactly one block per lower_key; a duplicate (e.g. a stale None
    placeholder beside the real block) can shadow the real value on a read.
    """
    low = omap._lower_oram
    tree = low._client._storage[low._name]
    counts = {}
    for i in range(tree.size):
        for data in tree.storage[i]:
            if data is not None and getattr(data, "key", None) is not None:
                counts[data.key] = counts.get(data.key, 0) + 1
    for data in low._stash:
        if data.key is not None:
            counts[data.key] = counts.get(data.key, 0) + 1
    return {key: c for key, c in counts.items() if c > 1}


class TestGroupOmap:
    @pytest.mark.parametrize("upper_kind", ["path", "da"])
    @pytest.mark.parametrize("enc", [False, True])
    def test_oracle(self, upper_kind, enc, client):
        n = 128
        upper = _make_upper(upper_kind, n, client, AesGcm() if enc else None)
        omap = GroupOmap(
            GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client, encryptor=AesGcm() if enc else None),
            upper_oram=upper,
        )
        omap.init_server_storage()
        rng = random.Random(hash((upper_kind, enc)) & 0xFFFF)
        model, keyspace = {}, list(range(n))
        for _ in range(n * 4):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.5:
                if key not in model:  # insert does not dedupe keys
                    value = rng.randint(0, 10**6)
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.75:
                assert omap.search(key=key) == model.get(key)
            elif key in model:  # update an existing key's value, checking the returned old value
                new_value = rng.randint(0, 10**6)
                assert omap.search(key=key, value=new_value) == model[key]
                model[key] = new_value
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        assert omap.search(key=99999) is None  # missing key
        if not enc:
            assert not _duplicate_blocks(omap)

    def test_undersized_upper_raises(self, client):
        n = 128
        too_small = PathOram(PathOramConfig(num_data=n, data_size=8, client=client, name="group_upper"))
        with pytest.raises(ValueError, match="too small"):
            GroupOmap(GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client), upper_oram=too_small)

    def test_with_init(self, client):
        n = 128
        upper = _make_upper("path", n, client, None)
        omap = GroupOmap(GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client), upper_oram=upper)
        init = [(i, i * 7) for i in range(n // 2)]
        omap.init_server_storage(data=init)
        model = dict(init)
        for i in range(n // 2, n):
            omap.insert(key=i, value=i * 7)
            model[i] = i * 7
        for key in range(n):
            assert omap.search(key=key) == model[key]
        assert not _duplicate_blocks(omap)
