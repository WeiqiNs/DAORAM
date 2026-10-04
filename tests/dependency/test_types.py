import pickle
from dataclasses import dataclass

import pytest

from oblivlib.dependency import AVLData, BPlusData, Data


@dataclass
class _Point:
    x: int


def test_field_tuple_pickle_byte_format():
    cases = [
        (Data(key=1, leaf=2, value=b"v"), (1, 2, b"v")),
        (Data(key=1, leaf=2, value=_Point(3)), (1, 2, _Point(3))),
        (
            AVLData(value=b"v", r_key="r", r_leaf=3, r_height=1, l_key="l", l_leaf=4, l_height=2),
            (b"v", "r", 3, 1, "l", 4, 2),
        ),
        (BPlusData(keys=[1, 2], values=[(10, 0), (20, 1)]), ([1, 2], [(10, 0), (20, 1)])),
    ]
    for obj, field_tuple in cases:
        assert obj.dump() == pickle.dumps(field_tuple)
        assert type(obj).load(obj.dump()) == obj


class TestData:
    def test_dump_pad_pads_to_exact_length_and_round_trips(self):
        for block in (
            Data(key=1, leaf=2, value=b"hi"),
            Data(key="k", leaf=0, value=[1, 2, 3]),
            Data(),
            Data(key=1, leaf=2, value=_Point(3)),
        ):
            padded = block.dump_pad(200)
            assert len(padded) == 200
            assert Data.load(padded) == block
            assert Data.load(block.dump_pad(len(block.dump()))) == block

    def test_dump_pad_too_short_raises(self):
        with pytest.raises(ValueError):
            Data(key=1, leaf=2, value=b"x" * 10).dump_pad(3)
