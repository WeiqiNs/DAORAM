import random

import pytest

from oblivlib.dependency import (
    AesGcm,
    AvlOmapCachedConfig,
    AvlOmapConfig,
    BPlusOmapCachedConfig,
    BPlusOmapConfig,
    DaOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
)
from oblivlib.omap import AVLOmap, AVLOmapCached, BPlusOmap, BPlusOmapCached, OramOstOmap, OramOstOmapConfig
from oblivlib.oram import DAOram, PathOram, RecursivePathOram


def _make_ods(kind, n, client, encryptor=None):
    if kind == "avl":
        return AVLOmap(AvlOmapConfig(num_data=n, key_size=10, data_size=10, client=client, encryptor=encryptor))
    if kind == "avl_cached":
        return AVLOmapCached(
            AvlOmapCachedConfig(num_data=n, key_size=10, data_size=10, client=client, encryptor=encryptor)
        )
    if kind == "bplus":
        return BPlusOmap(
            BPlusOmapConfig(order=40, num_data=n, key_size=10, data_size=10, client=client, encryptor=encryptor)
        )
    return BPlusOmapCached(
        BPlusOmapCachedConfig(order=40, num_data=n, key_size=10, data_size=10, client=client, encryptor=encryptor)
    )


def _make_oram(kind, n, client, encryptor=None):
    if kind == "da":
        return DAOram(DaOramConfig(num_data=n, data_size=20, client=client, encryptor=encryptor))
    if kind == "path":
        return PathOram(PathOramConfig(num_data=n, data_size=20, client=client, encryptor=encryptor))
    return RecursivePathOram(RecursiveOramConfig(num_data=n, data_size=20, client=client, encryptor=encryptor))


class TestOramOstOmapOracle:
    """Model oracle over ODS x ORAM: the composition must match a plain dict under interleaved
    insert / search, including searches of keys whose hash bucket is still empty."""

    @pytest.mark.parametrize("ods_kind", ["avl", "avl_cached", "bplus", "bplus_cached"])
    @pytest.mark.parametrize("oram_kind", ["da", "path", "recursive"])
    @pytest.mark.parametrize("key_str", [False, True])
    def test_oracle(self, ods_kind, oram_kind, key_str, client):
        n = 64
        omap = OramOstOmap(
            OramOstOmapConfig(num_data=n), ost=_make_ods(ods_kind, n, client), oram=_make_oram(oram_kind, n, client)
        )
        omap.init_server_storage()
        rng = random.Random(hash((ods_kind, oram_kind, key_str)) & 0xFFFF)
        model = {}
        keyspace = [(f"k{i}" if key_str else i) for i in range(n)]
        for _ in range(n * 4):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.5:
                if key not in model:  # the ODS insert does not handle duplicate keys
                    value = f"v{rng.randint(0, 10**6)}" if key_str else rng.randint(0, 10**6)
                    omap.insert(key=key, value=value)
                    model[key] = value
            else:
                assert omap.search(key=key) == model.get(key)
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)


class TestOramOstOmapInit:
    """Bulk-init via init_server_storage(data=...); the oracle above only exercises empty init."""

    @pytest.mark.parametrize("ods_kind", ["avl", "avl_cached", "bplus", "bplus_cached"])
    @pytest.mark.parametrize("oram_kind", ["da", "path", "recursive"])
    @pytest.mark.parametrize("key_str", [False, True])
    def test_with_init(self, ods_kind, oram_kind, key_str, client):
        n = 64
        omap = OramOstOmap(
            OramOstOmapConfig(num_data=n), ost=_make_ods(ods_kind, n, client), oram=_make_oram(oram_kind, n, client)
        )
        keys = [(f"k{i}" if key_str else i) for i in range(n)]
        values = [(f"v{i}" if key_str else i * 2) for i in range(n)]

        omap.init_server_storage(data=list(zip(keys[: n // 2], values[: n // 2], strict=True)))
        for key, value in zip(keys[n // 2 :], values[n // 2 :], strict=True):
            omap.insert(key=key, value=value)

        for key, value in zip(keys, values, strict=True):
            assert omap.search(key=key) == value

    @pytest.mark.parametrize("ods_kind", ["avl", "bplus"])
    def test_encryption_round_trip(self, ods_kind, client, encryptor):
        n = 64
        omap = OramOstOmap(
            OramOstOmapConfig(num_data=n),
            ost=_make_ods(ods_kind, n, client, encryptor=encryptor),
            oram=_make_oram("da", n, client, encryptor=AesGcm()),
        )
        omap.init_server_storage()
        for i in range(n):
            omap.insert(key=i, value=i)
        for i in range(n):
            assert omap.search(key=i) == i
