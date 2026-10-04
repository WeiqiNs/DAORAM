import math

import pytest

from oblivlib.dependency.load_bound import lambert_w


def test_lambert_w_satisfies_defining_equation():
    for x in (0.5, 1.0, 2.0, 10.0, 100.0):
        w = lambert_w(x)
        assert abs(w * math.exp(w) - x) < 1e-6


def test_lambert_w_rejects_below_lower_bound():
    with pytest.raises(ValueError):
        lambert_w(-1.0)
