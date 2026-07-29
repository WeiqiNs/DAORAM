from typing import Literal

import pytest

from oblivlib.dependency import (
    AesGcm,
    DaOramConfig,
    FreecursiveOramConfig,
    MulPathOramConfig,
    PathOramConfig,
    RecursiveOramConfig,
    StaticOramConfig,
)
from oblivlib.oram import DAOram, FreecursiveOram, MulPathOram, PathOram, RecursivePathOram, StaticOram


def _path(**kw):
    return PathOram(PathOramConfig(**kw))


def _static(**kw):
    return StaticOram(StaticOramConfig(**kw))


def _mul_path(**kw):
    return MulPathOram(MulPathOramConfig(**kw))


def _recursive(**kw):
    return RecursivePathOram(RecursiveOramConfig(**kw))


def _da(**kw):
    return DAOram(DaOramConfig(**kw))


def _freecursive(reset_method: Literal["prob", "hard"] = "prob", **kw):
    return FreecursiveOram(FreecursiveOramConfig(reset_method=reset_method, **kw))


ORAM_SPECS = [
    pytest.param(_path, id="path"),
    pytest.param(_static, id="static"),
    pytest.param(_mul_path, id="mul_path"),
    pytest.param(_recursive, id="recursive"),
    pytest.param(_da, id="da"),
    pytest.param(lambda **kw: _freecursive(reset_method="prob", **kw), id="freecursive_prob"),
    pytest.param(lambda **kw: _freecursive(reset_method="hard", **kw), id="freecursive_hard"),
]


@pytest.fixture(params=ORAM_SPECS)
def make_oram(request):
    return request.param


@pytest.fixture(params=["memory", "memory_enc", "file", "file_enc"])
def storage_kwargs(request, test_file):
    kwargs = {}
    if "enc" in request.param:
        kwargs["encryptor"] = AesGcm()
    if "file" in request.param:
        kwargs["filename"] = str(test_file)
    return kwargs
