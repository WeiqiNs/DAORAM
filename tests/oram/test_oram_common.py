"""Behavioral tests run against every ORAM via the make_oram fixture (see conftest.py)."""

import random

import pytest

from oblivlib.dependency import DaOramConfig, FreecursiveOramConfig, RecursiveOramConfig
from oblivlib.oram import DAOram, FreecursiveOram, RecursivePathOram


class TestOramCommon:
    def test_round_trip(self, make_oram, num_data, client, storage_kwargs):
        oram = make_oram(num_data=num_data, data_size=10, client=client, **storage_kwargs)
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_with_init(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage(data_map={i: i * 2 for i in range(num_data)})

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i * 2

    def test_random_queries(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for _ in range(num_data * 5):
            key = random.randint(0, num_data - 1)
            oram.operate_on_key(key=key, value=key * 2)
            assert oram.operate_on_key(key=key) == key * 2

    def test_repeated_same_key(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for value in range(num_data * 5):
            oram.operate_on_key(key=0, value=value)
            assert oram.operate_on_key(key=0) == value

    def test_operate_then_evict(self, make_oram, num_data, client):
        oram = make_oram(num_data=num_data, data_size=10, client=client)
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key_without_eviction(key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i


@pytest.mark.parametrize(("cls", "config_cls"), [(DAOram, DaOramConfig), (FreecursiveOram, FreecursiveOramConfig)])
def test_pos_map_emptied_after_compression(cls, config_cls, num_data, client):
    oram = cls(config_cls(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert oram._pos_map == {}


def test_recursive_pos_map_compressed(num_data, client):
    oram = RecursivePathOram(RecursiveOramConfig(num_data=num_data, data_size=10, client=client))
    oram.init_server_storage()
    assert len(oram._pos_map) <= 10
