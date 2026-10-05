"""Behavioral tests run against every ORAM via the make_oram fixture (see conftest.py)."""

import random

import pytest

from oblivlib.dependency import UNSET, DaOramConfig, FreecursiveOramConfig, RecursiveOramConfig
from oblivlib.oram import DAOram, FreecursiveOram, RecursivePathOram


class TestOramCommon:
    def test_round_trip(self, make_oram, num_data, client, storage_kwargs):
        oram = make_oram(num_data=num_data, data_size=10, client=client, **storage_kwargs)
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=b"%d" % i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == b"%d" % i

    def test_with_init(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage(data={i: b"%d" % (i * 2) for i in range(num_data)})

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == b"%d" % (i * 2)

    def test_random_queries(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for _ in range(num_data * 5):
            key = random.randint(0, num_data - 1)
            oram.operate_on_key(key=key, value=b"%d" % (key * 2))
            assert oram.operate_on_key(key=key) == b"%d" % (key * 2)

    def test_repeated_same_key(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for value in range(num_data * 5):
            oram.operate_on_key(key=0, value=b"%d" % value)
            assert oram.operate_on_key(key=0) == b"%d" % value

    def test_operate_then_evict(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key_without_eviction(key=i)
            oram.eviction_with_update_stash(key=i, value=b"%d" % i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == b"%d" % i

    def test_write_deferral_preserves_access_sequence(self, make_oram, assert_deferral_preserves_access):
        def workload(client):
            oram = make_oram(num_data=64, data_size=10, client=client)
            oram.init_server_storage()
            rng = random.Random(5)
            for step in range(60):
                oram.operate_on_key(key=rng.randrange(64), value=b"%d" % step if step % 2 else UNSET)
            oram.operate_on_key_without_eviction(key=3)
            oram.eviction_with_update_stash(key=3, value=b"1")

        rounds_off, rounds_on = assert_deferral_preserves_access(workload)
        assert rounds_on * 2 == rounds_off


@pytest.mark.parametrize(("cls", "config_cls"), [(DAOram, DaOramConfig), (FreecursiveOram, FreecursiveOramConfig)])
def test_pos_map_emptied_after_compression(cls, config_cls, num_data, client):
    oram = cls(config_cls(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert oram._pos_map == {}


def test_recursive_pos_map_compressed(num_data, client):
    oram = RecursivePathOram(RecursiveOramConfig(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert len(oram._pos_map) <= 10


@pytest.mark.parametrize(
    "make_counter_oram",
    [
        lambda client: DAOram(DaOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2)),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=4, reset_prob=0.5)
        ),
        lambda client: FreecursiveOram(
            FreecursiveOramConfig(num_data=256, data_size=10, client=client, num_ic=4, ic_length=2, reset_method="hard")
        ),
    ],
    ids=["da", "freecursive_prob", "freecursive_hard"],
)
def test_write_deferral_preserves_counter_resets(make_counter_oram, assert_deferral_preserves_access):
    def workload(client):
        oram = make_counter_oram(client)
        oram.init_server_storage()
        rng = random.Random(6)
        for step in range(120):
            oram.operate_on_key(key=rng.randrange(256), value=b"%d" % step)
        for key in range(4):
            oram.operate_on_key_without_eviction(key=key)
            oram.eviction_with_update_stash(key=key, value=b"%d" % key, execute=False)

    assert_deferral_preserves_access(workload)
