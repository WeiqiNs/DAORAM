import math
import pickle
from typing import Any

import pytest

from oblivlib.dependency import Blake2Prf, Helper


class TestHelper:
    def test_pad_pickle_pads_to_exact_length_and_round_trips(self):
        for obj in ((1, 2, b"hi"), ("k", 0, [1, 2, 3]), (None, None, None)):
            padded = Helper.pad_pickle(data=pickle.dumps(obj), length=200)
            assert len(padded) == 200
            assert pickle.loads(padded) == obj

    def test_pad_pickle_too_short_raises(self):
        with pytest.raises(ValueError):
            Helper.pad_pickle(data=b"\x00" * 10, length=3)

    def test_binary_str_conversion_round_trip(self):
        binary_str = "100100100"
        assert Helper.bytes_to_binary_str(binary_bytes=Helper.binary_str_to_bytes(binary_str=binary_str)) == binary_str

    def test_hash_data_to_leaf_in_range_and_deterministic(self):
        prf = Blake2Prf()
        for data in (42, "key", b"bytes"):
            first = Helper.hash_data_to_leaf(prf=prf, map_size=64, data=data)
            assert 0 <= first < 64
            assert Helper.hash_data_to_leaf(prf=prf, map_size=64, data=data) == first

    def test_hash_data_to_leaf_rejects_unsupported_type(self):
        unsupported: Any = 3.14
        with pytest.raises(TypeError):
            Helper.hash_data_to_leaf(prf=Blake2Prf(), map_size=64, data=unsupported)

    def test_lambert_w_satisfies_defining_equation(self):
        for x in (0.5, 1.0, 2.0, 10.0, 100.0):
            w = Helper.lambert_w(x)
            assert abs(w * math.exp(w) - x) < 1e-6

    def test_lambert_w_rejects_below_lower_bound(self):
        with pytest.raises(ValueError):
            Helper.lambert_w(-1.0)
