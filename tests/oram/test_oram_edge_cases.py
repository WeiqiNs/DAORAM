import random
import secrets

import pytest

from oblivlib.dependency import DaOramConfig, FreecursiveOramConfig, PathOramConfig, RecursiveOramConfig
from oblivlib.oram import DAOram, FreecursiveOram, PathOram, RecursivePathOram

RECURSIVE_POS_MAP_ORAMS = [
    (RecursivePathOram, RecursiveOramConfig),
    (DAOram, DaOramConfig),
    (FreecursiveOram, FreecursiveOramConfig),
]


@pytest.mark.parametrize("n", [100, 1023, 1025])
def test_non_power_of_two_round_trip(make_oram, client, n):
    oram = make_oram(num_data=n, data_size=10, client=client)
    oram.init_server_storage()

    for i in range(n):
        oram.operate_on_key(key=i, value=i)

    for i in range(n):
        assert oram.operate_on_key(key=i) == i


@pytest.mark.parametrize("n", [1, 2, 7, 100, 1000, 1023, 1024, 1025, 4096])
def test_leaf_range_is_smallest_power_of_two_at_least_n(client, n):
    oram = PathOram(PathOramConfig(num_data=n, data_size=10, client=client))
    assert n <= oram._leaf_range < 2 * n


def test_init_path_overflow_goes_to_stash(client):
    oram = PathOram(PathOramConfig(num_data=4, data_size=10, client=client, bucket_size=1))
    oram._pos_map = dict.fromkeys(range(4), 0)
    oram.init_server_storage(data_map={i: f"v{i}" for i in range(4)})

    assert [data.key for data in oram._stash] == [3]
    for i in range(4):
        assert oram.operate_on_key(key=i) == f"v{i}"


@pytest.mark.parametrize(("cls", "config_cls"), RECURSIVE_POS_MAP_ORAMS)
def test_num_data_at_most_on_chip_raises(cls, config_cls, client):
    with pytest.raises(ValueError):
        cls(config_cls(num_data=8, data_size=10, client=client))


def test_da_oram_reset_block_left_in_pos_map_stash(monkeypatch, client):
    monkeypatch.setattr(secrets, "randbelow", random.Random(9).randrange)
    oram = DAOram(
        DaOramConfig(num_data=1024, data_size=8, client=client, bucket_size=1, stash_scale=1000, prf_key=b"\x00" * 32)
    )
    model = {i: i for i in range(1024)}
    oram.init_server_storage(data_map=dict(model))

    workload = random.Random(9)
    for value in range(3000):
        key = workload.randrange(1024)
        assert oram.operate_on_key(key=key, value=value) == model[key]
        model[key] = value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_data": 0, "data_size": 10},
        {"num_data": 16, "data_size": 0},
        {"num_data": 16, "data_size": 10, "bucket_size": 0},
        {"num_data": 16, "data_size": 10, "stash_scale": 0},
    ],
)
def test_config_rejects_invalid_numeric_fields(kwargs):
    with pytest.raises(ValueError):
        PathOramConfig(**kwargs)


@pytest.mark.parametrize("reset_prob", [0.0, -0.1, 1.5])
def test_freecursive_config_rejects_bad_reset_prob(reset_prob):
    with pytest.raises(ValueError):
        FreecursiveOramConfig(num_data=64, data_size=10, reset_prob=reset_prob)


def test_write_returns_old_value_and_none_is_writable(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage()

    oram.operate_on_key(key=3, value=42)
    assert oram.operate_on_key(key=3, value=99) == 42
    assert oram.operate_on_key(key=3) == 99
    assert oram.operate_on_key(key=3, value=None) == 99
    assert oram.operate_on_key(key=3) is None


def test_arbitrary_value_types(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage()

    values = {0: "hello", 1: b"\x00\x02\x04", 2: (1, 2, 3), 3: -7, 4: ["a", 1]}
    for key, value in values.items():
        oram.operate_on_key(key=key, value=value)

    for key, value in values.items():
        assert oram.operate_on_key(key=key) == value
