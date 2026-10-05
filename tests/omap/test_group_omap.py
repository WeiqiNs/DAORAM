"""Tests for GroupOmap (group-by-hash OMAP: upper ORAM for metadata + MulPathOram lower ORAM)."""

import random
from typing import Any

import pytest

from oblivlib.dependency import AesGcm, ContractError, DaOramConfig, GroupOmapConfig, PathOramConfig
from oblivlib.omap import GroupOmap
from oblivlib.oram import DAOram, PathOram


def _make_upper(kind, n, client, encryptor, key_size=10):
    data_size = GroupOmap.upper_oram_data_size(num_data=n, key_size=key_size)
    if kind == "path":
        return PathOram(
            PathOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor)
        )
    return DAOram(DaOramConfig(num_data=n, data_size=data_size, client=client, name="group_upper", encryptor=encryptor))


def _make_omap(client, n=128, encryptor=None, upper_kind="path"):
    return GroupOmap(
        GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client, encryptor=encryptor),
        upper_oram=_make_upper(upper_kind, n, client, encryptor),
    )


def _duplicate_blocks(omap, read_all_blocks):
    """Return {key: count} for any key with more than one block in the lower ORAM.

    The group design relies on exactly one block per key; a duplicate can shadow the real value on a read.
    """
    low = omap._lower_oram
    counts = {}
    for data in read_all_blocks(low) + low._stash:
        counts[data.key] = counts.get(data.key, 0) + 1
    return {key: c for key, c in counts.items() if c > 1}


class TestGroupOmap:
    @pytest.mark.parametrize("upper_kind", ["path", "da"])
    @pytest.mark.parametrize("enc", [False, True])
    def test_oracle(self, upper_kind, enc, client, read_all_blocks):
        n = 128
        omap = _make_omap(client, n, AesGcm() if enc else None, upper_kind)
        omap.init_server_storage()
        rng = random.Random(hash((upper_kind, enc)) & 0xFFFF)
        model, keyspace = {}, [b"%010d" % i for i in range(n)]
        for _ in range(n * 4):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.5:
                if key not in model:
                    value = rng.randbytes(64)
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.75:
                assert omap.search(key=key) == model.get(key)
            elif key in model:
                new_value = rng.randbytes(64)
                assert omap.search(key=key, value=new_value) == model[key]
                model[key] = new_value
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        assert omap.search(key=b"absent") is None
        assert not _duplicate_blocks(omap, read_all_blocks)

    def test_undersized_upper_raises(self, client):
        n = 128
        too_small = PathOram(PathOramConfig(num_data=n, data_size=8, client=client, name="group_upper"))
        with pytest.raises(ValueError, match="too small"):
            GroupOmap(GroupOmapConfig(num_data=n, key_size=10, data_size=64, client=client), upper_oram=too_small)

    def test_with_init(self, client, read_all_blocks):
        n = 128
        omap = _make_omap(client, n)
        init = [(b"%d" % i, b"%d" % (i * 7)) for i in range(n // 2)]
        omap.init_server_storage(data=init)
        model = dict(init)
        for i in range(n // 2, n):
            omap.insert(key=b"%d" % i, value=b"%d" % (i * 7))
            model[b"%d" % i] = b"%d" % (i * 7)
        for key, value in model.items():
            assert omap.search(key=key) == value
        assert not _duplicate_blocks(omap, read_all_blocks)

    def test_many_keys_all_read_back(self, client):
        n = 128
        omap = _make_omap(client, n)
        keys = [b"key-%03d" % i for i in range(60)]
        expected = {key: b"init-" + key for key in keys[:30]} | {key: b"insert-" + key for key in keys[30:]}
        omap.init_server_storage(data=[(key, expected[key]) for key in keys[:30]])
        for key in keys[30:]:
            omap.insert(key=key, value=expected[key])

        assert {key: omap.search(key=key) for key in keys} == expected

    def test_rejects_contract_violations(self, client):
        omap = _make_omap(client)
        omap.init_server_storage(data=[(b"k", b"kept")])
        rounds = client.metrics.rounds

        cases: list[tuple[Any, Any]] = [("k", None), (b"x" * 11, None), (b"k", "v"), (b"k", b"x" * 65)]
        for key, value in cases:
            with pytest.raises(ContractError, match="GroupOmap 'group_omap'"):
                omap.search(key=key, value=value)
            with pytest.raises(ContractError, match="GroupOmap 'group_omap'"):
                omap.insert(key=key, value=value if value is not None else b"v")

        assert client.metrics.rounds == rounds
        assert omap.search(key=b"k") == b"kept"

    def test_write_deferral_preserves_access_sequence(self, assert_deferral_preserves_access):
        def workload(client):
            omap = _make_omap(client, 64)
            omap.init_server_storage(data=[(b"%d" % i, b"%d" % i) for i in range(0, 64, 2)])
            for i in range(1, 20, 2):
                omap.insert(key=b"%d" % i, value=b"%d" % i)
            for i in range(0, 30, 3):
                omap.search(key=b"%d" % i, value=b"%d" % (i + 1))

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on * 2 == rounds_off
