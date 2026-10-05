import pytest

from oblivlib.dependency import (
    Data,
    MissingClientError,
    PathOramConfig,
    StashOverflowError,
)
from oblivlib.dependency.tree_storage_base import TreeStorageBase


def test_missing_client_raises_identified_error():
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, name="orphan"))
    with pytest.raises(MissingClientError, match="TreeStorageBase 'orphan' has no client"):
        _ = base._client


def test_stash_overflow_raises():
    base = TreeStorageBase(PathOramConfig(num_data=4, data_size=4, stash_scale=1, name="tiny"))
    base._stash = [Data(key=i, leaf=0) for i in range(base._stash_capacity)]
    base._check_stash()

    base._stash.append(Data(key=99, leaf=0))
    with pytest.raises(StashOverflowError, match="TreeStorageBase 'tiny'") as raised:
        base._check_stash()
    assert isinstance(raised.value, MemoryError)
