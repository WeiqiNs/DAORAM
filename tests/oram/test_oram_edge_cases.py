import random
import secrets
from typing import Any

import pytest

from oblivlib.dependency import (
    ContractError,
    DaOramConfig,
    FreecursiveOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
)
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
        oram.operate_on_key(key=i, value=b"%d" % i)

    for i in range(n):
        assert oram.operate_on_key(key=i) == b"%d" % i


@pytest.mark.parametrize("n", [1, 2, 7, 100, 1000, 1023, 1024, 1025, 4096])
def test_leaf_range_is_smallest_power_of_two_at_least_n(client, n):
    oram = PathOram(PathOramConfig(num_data=n, data_size=10, client=client))
    assert n <= oram._leaf_range < 2 * n


def test_init_path_overflow_goes_to_stash(client):
    oram = PathOram(PathOramConfig(num_data=4, data_size=10, client=client, bucket_size=1))
    oram._pos_map = dict.fromkeys(range(4), 0)
    oram.init_server_storage(data={i: b"v%d" % i for i in range(4)})

    assert [data.key for data in oram._stash] == [3]
    for i in range(4):
        assert oram.operate_on_key(key=i) == b"v%d" % i


def test_init_accepts_one_shot_stream(make_oram, client, test_file, encryptor):
    oram = make_oram(num_data=64, data_size=10, client=client, encryptor=encryptor, build_file=test_file)
    oram.init_server_storage(data=((key, b"v%d" % key) for key in range(64) if key % 3))

    assert not test_file.exists()
    for key in range(64):
        assert oram.operate_on_key(key=key) == (b"v%d" % key if key % 3 else b"")


@pytest.mark.parametrize(
    ("pairs", "error", "match"),
    [
        ([(1, b"a"), (1, b"b")], ValueError, "given twice"),
        ([(64, b"a")], ContractError, r"\[0, 64\)"),
        ([(-1, b"a")], ContractError, r"\[0, 64\)"),
        ([(1, "a")], ContractError, "must be bytes"),
        ([(1, b"x" * 11)], ContractError, "data_size is 10"),
    ],
)
def test_init_rejects_duplicate_keys_and_contract_violations(client, pairs, error, match):
    oram = PathOram(PathOramConfig(num_data=64, data_size=10, client=client))
    with pytest.raises(error, match=match):
        oram.init_server_storage(data=iter(pairs))


@pytest.mark.parametrize(("cls", "config_cls"), RECURSIVE_POS_MAP_ORAMS)
def test_num_data_at_most_on_chip_raises(cls, config_cls, client):
    with pytest.raises(ValueError):
        cls(config_cls(num_data=8, data_size=10, client=client))


def test_da_oram_reset_block_left_in_pos_map_stash(monkeypatch, client):
    monkeypatch.setattr(secrets, "randbelow", random.Random(9).randrange)
    oram = DAOram(
        DaOramConfig(num_data=1024, data_size=8, client=client, bucket_size=1, stash_scale=1000, prf_key=b"\x00" * 32)
    )
    model = {i: b"%d" % i for i in range(1024)}
    oram.init_server_storage(data=dict(model))

    workload = random.Random(9)
    for step in range(3000):
        key = workload.randrange(1024)
        assert oram.operate_on_key(key=key, value=b"%d" % step) == model[key]
        model[key] = b"%d" % step


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


def test_config_rejects_plaintext_build_file(test_file):
    with pytest.raises(ValueError, match="build_file requires an encryptor"):
        PathOramConfig(num_data=16, data_size=10, build_file=test_file)


@pytest.mark.parametrize("reset_prob", [0.0, -0.1, 1.5])
def test_freecursive_config_rejects_bad_reset_prob(reset_prob):
    with pytest.raises(ValueError):
        FreecursiveOramConfig(num_data=64, data_size=10, reset_prob=reset_prob)


def test_write_returns_old_value_and_empty_bytes_is_writable(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage()

    assert oram.operate_on_key(key=3) == b""
    oram.operate_on_key(key=3, value=b"42")
    assert oram.operate_on_key(key=3, value=b"99") == b"42"
    assert oram.operate_on_key(key=3) == b"99"
    assert oram.operate_on_key(key=3, value=b"") == b"99"
    assert oram.operate_on_key(key=3) == b""


def test_rejects_non_bytes_oversized_and_out_of_range(make_oram, client):
    oram = make_oram(num_data=64, data_size=10, client=client)
    oram.init_server_storage()
    oram.operate_on_key(key=5, value=b"kept")
    rounds = client.metrics.rounds

    cases: list[tuple[Any, Any]] = [
        (5, "hello"),
        (5, b"x" * 11),
        (5, None),
        (-1, b"v"),
        (64, b"v"),
        ("0", b"v"),
        (True, b"v"),
    ]
    for key, value in cases:
        with pytest.raises(ContractError, match=f"{type(oram).__name__} '"):
            oram.operate_on_key(key=key, value=value)

    assert client.metrics.rounds == rounds
    assert oram.operate_on_key(key=5) == b"kept"
