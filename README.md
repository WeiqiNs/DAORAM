# oblivlib: oblivious algorithms in Python

`oblivlib` implements classic oblivious algorithms (oblivious RAM and oblivious maps) for client/server
deployments where the client is trusted and may compute non-obliviously, and the server only stores
encrypted data. Design notes are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

Related publications by the maintainers:

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

## Usage

```bash
pip install .            # or: pip install -e ".[dev]" for pytest, basedpyright, ruff
```

```python
from oblivlib.dependency import Client, PathOramConfig
from oblivlib.oram import PathOram

with Client.local() as client:
    oram = PathOram(PathOramConfig(num_data=1024, data_size=16, client=client))
    oram.init_server_storage()
    oram.operate_on_key(3, b"hello")
    assert oram.operate_on_key(3) == b"hello"
```

ORAM keys are int addresses in `[0, num_data)` and OMAP keys are `bytes` of at most `key_size`; every
value is `bytes` of at most `data_size`, so serialize your own data before storing it. A key or value
outside that contract raises `ContractError` before anything touches the server.

`Client.local()` keeps the server in process. For a remote deployment, run
[`demo/server.py`](demo/server.py) on the server and connect with `Client.connect("tcp://host:5555")`, as
[`demo/oram_client.py`](demo/oram_client.py) and [`demo/omap_client.py`](demo/omap_client.py) do. The
shared test suites ([`tests/oram/test_oram_common.py`](tests/oram/test_oram_common.py),
[`tests/omap/test_omap_common.py`](tests/omap/test_omap_common.py)) show every scheme in use.
