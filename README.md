# oblivlib: oblivious algorithms in Python

`oblivlib` implements classic oblivious algorithms (oblivious RAM, oblivious maps, and oblivious graph
processing) for client/server deployments where the client is trusted and may compute non-obliviously,
and the server only stores encrypted data. Design notes are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

Related publications by the maintainers:

- Enabling Index-free Adjacency in Oblivious Graph Processing with Delayed Duplications
- [Towards Practical Oblivious Map (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235)

## ORAM

- [Path ORAM (CCS 2013)](https://dl.acm.org/doi/10.1145/2508859.2516660):
  [`path_oram.py`](src/oblivlib/oram/path_oram.py)
- [Recursive Path ORAM (ASIACRYPT 2011)](https://link.springer.com/chapter/10.1007/978-3-642-25385-0_11):
  [`recursive_path_oram.py`](src/oblivlib/oram/recursive_path_oram.py)
- [Freecursive ORAM (ASPLOS 2015)](https://people.csail.mit.edu/devadas/pubs/freecursive.pdf), insecure
  original: [`freecursive_oram.py`](src/oblivlib/oram/freecursive_oram.py) with `reset_method="hard"`
- [Freecursive ORAM with probabilistic resets (TCC 2017)](https://eprint.iacr.org/2016/1084):
  [`freecursive_oram.py`](src/oblivlib/oram/freecursive_oram.py) with `reset_method="prob"` (default)
- [DAORAM, de-amortized fixed resets (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235):
  [`da_oram.py`](src/oblivlib/oram/da_oram.py)
- Multi-path and static (PRF-positioned) Path ORAM variants:
  [`mul_path_oram.py`](src/oblivlib/oram/mul_path_oram.py), [`static_oram.py`](src/oblivlib/oram/static_oram.py)

## OMAP

- [AVL-tree OMAP (CCS 2014)](https://dl.acm.org/doi/10.1145/2660267.2660314) with the streaming search of
  [VLDB 2024](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf): [`avl_omap.py`](src/oblivlib/omap/avl_omap.py)
- [B+-tree OMAP (VLDB 2020)](https://people.eecs.berkeley.edu/~matei/papers/2020/vldb_oblidb.pdf):
  [`bplus_omap.py`](src/oblivlib/omap/bplus_omap.py)
- [ORAM + search-tree framework (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235), composing
  any ORAM with any tree OMAP: [`oram_ost_omap.py`](src/oblivlib/omap/oram_ost_omap.py)
- Group-by-hash OMAP: [`group_omap.py`](src/oblivlib/omap/group_omap.py)
- Cache-optimized AVL and B+ variants (not per-op oblivious):
  [`avl_omap_cache.py`](src/oblivlib/omap/avl_omap_cache.py), [`bplus_omap_cache.py`](src/oblivlib/omap/bplus_omap_cache.py)

Tree OMAPs hide the operation type by default (`distinguishable=False`); set it to `True` to let each
operation pad only to its own bound.

## Graph

- [GraphOS (VLDB 2024)](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf):
  [`graphos.py`](src/oblivlib/graph/graphos.py)
- Grove, with delayed duplications: [`grove.py`](src/oblivlib/graph/grove.py)

The graph and SORAM modules have not yet been ported to the current API.

## Usage

```bash
pip install .            # or: pip install -e ".[dev]" for pytest, basedpyright, ruff
```

```python
from oblivlib.dependency import InteractLocalServer, PathOramConfig
from oblivlib.oram import PathOram

oram = PathOram(PathOramConfig(num_data=1024, data_size=16, client=InteractLocalServer()))
oram.init_server_storage()
oram.operate_on_key(3, b"hello")
assert oram.operate_on_key(3) == b"hello"
```

For a remote deployment, start [`demo/server.py`](demo/server.py) on the server and run
[`demo/oram_client.py`](demo/oram_client.py) or [`demo/omap_client.py`](demo/omap_client.py) on the
client. The shared test suites ([`tests/oram/test_oram_common.py`](tests/oram/test_oram_common.py),
[`tests/omap/test_omap_common.py`](tests/omap/test_omap_common.py)) show every scheme in use.
