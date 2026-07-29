"""Shared fixtures and helpers for the OMAP test suite.

``omap_spec`` parametrizes the behavioral suite (``test_omap_common.py``) over every ODS OMAP, so a
single oracle/invariant/obliviousness suite runs against all of them. Scheme-specific tests stay in
their own ``test_<scheme>.py`` files. New schemes (composed maps) are added to ``OMAP_SPECS``.

Each spec records the behavioral guarantees the suite holds the scheme to:
  * per_op_oblivious -- search/insert induce a fixed-shape access pattern (cached variants do not);
  * delete_oblivious -- delete does too (false where delete leaks structure, e.g. non-cached B+);
  * search_none      -- search(None) is a supported dummy that must match a real search.
"""

from typing import override

import pytest

from oblivlib.dependency import (
    AvlOmapCachedConfig,
    AvlOmapConfig,
    BPlusOmapCachedConfig,
    BPlusOmapConfig,
    InteractLocalServer,
)
from oblivlib.omap import AVLOmap, AVLOmapCached, BPlusOmap, BPlusOmapCached


class CountingServer(InteractLocalServer):
    """Local server that counts path-read rounds, so tests can assert access-pattern uniformity."""

    def __init__(self):
        super().__init__()
        self.reads = 0

    @override
    def add_read_path(self, label, leaves):
        self.reads += len(leaves)
        return super().add_read_path(label, leaves)

    def rounds(self, fn, *args, **kwargs):
        """Run fn and return (number of path-read rounds it triggered, fn's result)."""
        before = self.reads
        result = fn(*args, **kwargs)
        return self.reads - before, result


def _live_blocks(omap):
    """All live ORAM blocks {block_key: Data} from storage + stash + local (plaintext only)."""
    tree = omap._client._storage[omap._name]
    blocks = {}
    for index in range(tree.size):
        for data in tree.storage[index]:
            if data is not None and getattr(data, "key", None) is not None:
                blocks[data.key] = data
    # The freshest copy of a block lives in stash/local, not the (stale) server path it was read from.
    for data in list(omap._stash) + omap._local.to_list():
        if data.key is not None:
            blocks[data.key] = data
    return blocks


def verify_avl_invariants(omap, model):
    """Assert BST order, AVL balance, stored-height accuracy, and parent->child leaf links.

    ``model`` is the key->value oracle; the AVL check only needs its key set. Plaintext only.
    """
    nodes = _live_blocks(omap)
    assert set(nodes) == set(model), f"live keys mismatch: have {len(nodes)}, expected {len(model)}"
    if omap.root is None:
        assert not model
        return

    def walk(key, leaf, low, high):
        assert key in nodes, f"node {key} referenced by a pointer but not present"
        data = nodes[key]
        assert data.leaf == leaf, f"node {key}: stored leaf {data.leaf} != parent pointer {leaf}"
        assert (low is None or low < key) and (high is None or key < high), (
            f"BST order violated at {key} (bounds {low}, {high})"
        )
        value = data.value
        l_height = walk(value.l_key, value.l_leaf, low, key) if value.l_key is not None else 0
        r_height = walk(value.r_key, value.r_leaf, key, high) if value.r_key is not None else 0
        assert value.l_height == l_height, f"node {key}: stored l_height {value.l_height} != {l_height}"
        assert value.r_height == r_height, f"node {key}: stored r_height {value.r_height} != {r_height}"
        assert abs(l_height - r_height) <= 1, f"node {key} unbalanced (l={l_height}, r={r_height})"
        return 1 + max(l_height, r_height)

    root_key, root_leaf = omap.root
    walk(root_key, root_leaf, None, None)


def verify_bplus_invariants(omap, model):
    """Assert parent->child leaf links, key routing/order, and that leaves hold exactly ``model``.

    ``model`` is the key->value oracle; the B+ leaves must reproduce it exactly. Plaintext only.
    """
    if omap.root is None:
        assert not model
        return
    blocks = _live_blocks(omap)
    collected = {}

    def is_leaf(value):
        return len(value.keys) == len(value.values)

    def walk(block_id, leaf, low, high):
        assert block_id in blocks, f"block {block_id} referenced by a pointer but not present"
        data = blocks[block_id]
        assert data.leaf == leaf, f"block {block_id}: stored leaf {data.leaf} != pointer {leaf}"
        value = data.value
        assert list(value.keys) == sorted(value.keys), f"keys unsorted in block {block_id}: {value.keys}"
        for k in value.keys:
            assert (low is None or low <= k) and (high is None or k < high), (
                f"key {k} outside routing bounds ({low}, {high}) in block {block_id}"
            )
        if is_leaf(value):
            for k, v in zip(value.keys, value.values, strict=True):
                assert k not in collected, f"duplicate key {k} across leaves"
                collected[k] = v
        else:
            assert len(value.values) == len(value.keys) + 1, f"internal arity off in block {block_id}"
            low_i = low
            for i, (child_id, child_leaf) in enumerate(value.values):
                high_i = value.keys[i] if i < len(value.keys) else high
                walk(child_id, child_leaf, low_i, high_i)
                low_i = value.keys[i] if i < len(value.keys) else low_i

    root_id, root_leaf = omap.root
    walk(root_id, root_leaf, None, None)
    assert collected == model, (
        f"leaves != model (missing {set(model) - set(collected)}, extra {set(collected) - set(model)})"
    )


class OmapSpec:
    """Builds one OMAP scheme and records the behavioral guarantees the suite should hold it to."""

    def __init__(
        self,
        cls,
        config_cls,
        *,
        cached,
        supports_delete,
        per_op_oblivious,
        delete_oblivious,
        search_none,
        invariant_checker,
        extra=None,
    ):
        self.cls = cls
        self.config_cls = config_cls
        self.cached = cached
        self.supports_delete = supports_delete
        self.per_op_oblivious = per_op_oblivious
        self.delete_oblivious = delete_oblivious
        self.search_none = search_none
        self.invariant_checker = invariant_checker
        self.extra = extra or {}

    def make(self, *, client, num_data, key_size=16, data_size=16, **kw):
        config_kw = {**self.extra, **kw}
        return self.cls(
            self.config_cls(num_data=num_data, key_size=key_size, data_size=data_size, client=client, **config_kw)
        )


OMAP_SPECS = [
    pytest.param(
        OmapSpec(
            AVLOmap,
            AvlOmapConfig,
            cached=False,
            supports_delete=True,
            per_op_oblivious=True,
            delete_oblivious=True,
            search_none=False,
            invariant_checker=verify_avl_invariants,
        ),
        id="avl",
    ),
    pytest.param(
        OmapSpec(
            AVLOmapCached,
            AvlOmapCachedConfig,
            cached=True,
            supports_delete=True,
            per_op_oblivious=False,
            delete_oblivious=False,
            search_none=False,
            invariant_checker=verify_avl_invariants,
        ),
        id="avl_cached",
    ),
    pytest.param(
        # Non-cached B+ delete pre-fetches a sibling per level, so every path reads a fixed 2h-1
        # rounds regardless of borrow/merge -- oblivious, like search/insert.
        OmapSpec(
            BPlusOmap,
            BPlusOmapConfig,
            cached=False,
            supports_delete=True,
            per_op_oblivious=True,
            delete_oblivious=True,
            search_none=True,
            invariant_checker=verify_bplus_invariants,
            extra={"order": 5},
        ),
        id="bplus",
    ),
    pytest.param(
        OmapSpec(
            BPlusOmapCached,
            BPlusOmapCachedConfig,
            cached=True,
            supports_delete=True,
            per_op_oblivious=False,
            delete_oblivious=False,
            search_none=True,
            invariant_checker=verify_bplus_invariants,
            extra={"order": 5},
        ),
        id="bplus_cached",
    ),
]


@pytest.fixture(params=OMAP_SPECS)
def omap_spec(request):
    return request.param


def _specs_where(predicate):
    """The OMAP_SPECS params (keeping their ids) whose OmapSpec satisfies predicate."""
    return [param for param in OMAP_SPECS if predicate(param.values[0])]


@pytest.fixture(params=_specs_where(lambda s: s.per_op_oblivious))
def oblivious_omap_spec(request):
    """Only the variants that claim per-op obliviousness (cached variants are excluded by design, so
    the obliviousness suite collects exactly the variants it applies to -- no runtime skips)."""
    return request.param


@pytest.fixture(params=_specs_where(lambda s: s.delete_oblivious))
def delete_oblivious_omap_spec(request):
    """Only the variants whose delete is per-op oblivious."""
    return request.param


@pytest.fixture
def counting_server_cls():
    """The CountingServer class, so tests can spin up fresh access-pattern-counting servers."""
    return CountingServer
