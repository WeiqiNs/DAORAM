# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`oblivlib` is a Python library implementing classic oblivious algorithms — Oblivious RAM (ORAM), Oblivious Map (OMAP), and oblivious graph processing. It targets a client/server deployment where the client is trusted and may compute non-obliviously, while the server only stores encrypted data and answers batched read/write queries. The library backs the maintainers' publications (notably "Towards Practical Oblivious Map", VLDB 2025, and the Grove graph-processing work). See `README.md` for the paper-to-file mapping of each algorithm.

**For the full design knowledge base — data flow, the storage/ORAM/OMAP layers, the position-map and counter schemes, eviction, invariants/contracts, the test architecture, and known issues — read `ARCHITECTURE.md`.** This file is a quick reference; `ARCHITECTURE.md` is the deep one.

## Commands

- Run all tests: `pytest` (the `soram/` test dir is excluded by default via `collect_ignore` in `tests/conftest.py`). All of `tests/dependency`, `tests/oram`, and `tests/omap` collect and pass.
- Run a single test file: `pytest tests/oram/test_oram_common.py`
- Run a single test: `pytest "tests/oram/test_oram_common.py::TestOramCommon::test_round_trip"`
- Control dataset size in tests: the `num_data` fixture defaults to `2**12` (`DEFAULT_NUM_DATA` in `tests/conftest.py`); override a single run via the `NUM_DATA` env var — smaller for a fast local loop, larger to stress — e.g. `NUM_DATA=256 pytest tests/oram/test_oram_common.py`.
- Run soram tests (excluded by default): `pytest tests/soram/`
- Install for development: `pip install -e ".[dev]"` (runtime deps plus `pytest`, `basedpyright`, and `ruff`). All project metadata and dependencies live in `pyproject.toml` — runtime under `[project.dependencies]`, dev tooling under the `dev` optional-dependencies extra.
- Type-check: `basedpyright` (config in `pyrightconfig.json`, `recommended` mode — the same checker Zed runs, so editor and CI agree). `recommended` is `standard` plus basedpyright's stricter checks; the dynamic-typing rules (`reportUnknown*`, `reportAny`, `reportMissingParameterType`, etc.) are explicitly disabled in the config because this library is generic over arbitrary data (`Data.key`/`value` are `Any`, values are pickled), so `Any`/`Unknown` is pervasive by design — not because of the deps, which are fully typed. See the comment block there. The package is type-clean across `dependency`/`oram`/`omap`; `soram` and `graph` are excluded — both are broken against the current API and await a port (see `ARCHITECTURE.md` §10). basedpyright reads `pyrightconfig.json` and must be pointed at the env where the deps are installed (via `--pythonpath` or your editor's interpreter selection); otherwise every third-party import fails to resolve and cascades into spurious errors.
- Lint/format: `ruff check` and `ruff format --check` (config under `[tool.ruff]` in `pyproject.toml`).

CI (`.github/workflows/ci.yml`) installs `.[dev]` and runs `ruff check`, `ruff format --check`, `basedpyright`, and `pytest` on push/PR to `main`.

## Architecture

### Client/server split
The central abstraction is `InteractServer` (`src/oblivlib/dependency/interact_server.py`). All ORAM/OMAP/graph constructions talk to storage only through this interface — they never touch storage directly. Three implementations:
- `InteractLocalServer` — storage lives in the same process; used by all tests and for local development.
- `InteractRemoteServer` — client side; serializes batched queries and sends them over a socket.
- `RemoteServer` — server side; receives requests, runs them against local storage, returns results.

Construct a scheme with a `client` (an `InteractServer`), call `init_server_storage(...)` to populate storage, then issue operations. Swapping local for remote is transparent to the algorithm code.

### Query batching model
Operations do not perform I/O immediately. A construction accumulates queries on the client via `add_read_path` / `add_write_path` / `add_read_block` / `add_read_list` etc. (each keyed by a string `label` identifying the storage), then calls `client.execute()`, which runs all pending **writes first, then reads**, and returns an `ExecuteResult` (`success`, `results` dict keyed by label, `error`); schemes read a result via `result.require(label)`, which raises the underlying error on a failed `execute()` instead of masking it as a `KeyError`. The server also tracks bandwidth (`get_bandwidth`, `reset_bandwidth`) for experiments. Multiple ORAMs/OMAPs sharing one client must use distinct `name`/`label` values so their storage doesn't collide.

### Storage and data types
- Server storage is `Dict[str, Union[BinaryTree, List]]` — each label maps to either a `BinaryTree` (path-ORAM tree) or a plain list.
- `BinaryTree` (`src/oblivlib/dependency/binary_tree.py`) holds buckets of blocks; it has static helpers (`get_mul_path_dict`, `fill_data_to_path`, the O(1) closed-form `get_cross_index`) used by eviction logic. It delegates to `Storage` (`storage.py`), a thin facade over `_MemoryBackend` / `_FileBackend`. Encryption is **per bucket** (one ciphertext per bucket, via `Helper.encrypt_bucket`/`decrypt_bucket`); the file backend is two-phase (plaintext block-slots during fill, then a streaming `encrypt()` seal). See `ARCHITECTURE.md` §3/§8.
- Core data classes and type aliases live in `src/oblivlib/dependency/types.py` and `helper.py`: `Data` (a block: `key`, `leaf`, `value`), `PathData`/`BucketData`/`BlockData` (query payloads), `BucketKey`/`BlockKey` (NamedTuple indices), and the `UNSET` sentinel used to distinguish "read" from "write a value of `None`".

### ORAM layer (`src/oblivlib/oram/`)
Every scheme is constructed from a single frozen, keyword-only **config object** (defaults centralized in `src/oblivlib/dependency/config.py`), e.g. `PathOram(PathOramConfig(num_data=.., data_size=.., client=..))`; the config hierarchy mirrors the scheme hierarchy (`OramConfig` → `Da/Freecursive/Recursive/MulPath...Config`, `OmapConfig` adds `key_size`). See `ARCHITECTURE.md` §3 "Construction configs" (configs validate in `__post_init__`). `TreeBaseOram` (`tree_base_oram.py`) is the abstract base for all tree-based ORAMs; it and the OMAP `OstBaseOmap` both inherit `TreeStorageBase` (`src/oblivlib/dependency/tree_storage_base.py`), which owns the shared config accessors, leaf math, stash, and path encryption/decryption. `TreeBaseOram` adds the position map and stash eviction (`_evict_stash`, `_retrieve_data_block`). Subclasses implement `init_server_storage`, `operate_on_key`, `operate_on_key_without_eviction`, and `eviction_with_update_stash`. The key public operation is `operate_on_key(key, value=UNSET)` — always returns the current value, and writes `value` if one is provided. Concrete schemes: `PathOram`, `RecursivePathOram`, `FreecursiveOram` (insecure with `reset_method="hard"`, secure with `"prob"`), `DAOram`, `MulPathOram`, `StaticOram`.

### OMAP layer (`src/oblivlib/omap/`)
Two families:
- **Oblivious Search Tree (ODS)** maps, based on `OstBaseOmap` (`ost_base_omap.py`) — an abstract base that, like `TreeBaseOram`, inherits `TreeStorageBase`, but maintains a `root`, `local`, and `stash`. It exposes `insert`, `search` (the streaming descent that absorbed the old `fast_search`), and `delete`. Concrete: `AVLOmap`, `BPlusOmap`, and cache-optimized `AVLOmapCached` / `BPlusOmapCached`. The `distinguishable` config flag (default `False`) trades op-type hiding for speed; the cached variants are deliberately not per-op oblivious (see `ARCHITECTURE.md` §5).
- **Composed maps**: `OramOstOmap` (`oram_ost_omap.py`) is the VLDB 2025 framework — it combines *any* `TreeBaseOram` with *any* `OstBaseOmap`, hashing keys via a PRF into the ORAM. `GroupOmap` uses a group-by-hash design with an upper ORAM for metadata and a `MulPathOram` lower ORAM for key-value pairs.

### Graph layer (`src/oblivlib/graph/`)
`GraphOS` (GraphOS, VLDB 2024) and `Grove` (delayed-duplications scheme) both build on `MulPathOram` plus `AVLOmapCached`, using `src/oblivlib/dependency/graph.py` for the graph representation.

### SORAM (`src/oblivlib/soram/`)
A weaker-threat-model ORAM (protects access patterns only over windows of `c` consecutive operations) — more efficient than full ORAM. Tests are excluded from the default `pytest` run.

### Crypto (`src/oblivlib/dependency/crypto.py`)
`Encryptor` is the abstract encryption interface (`AesGcm` is the AES-GCM implementation), alongside `PRP` (`FeistelPrp`) and PRF (`Blake2Prf`) primitives. Constructions take an `encryptor` argument; when `None`, data is stored in plaintext (paths skip the encrypt/decrypt step) — useful for debugging. Buckets are encrypted as a unit (per-bucket ciphertext); GCM auth is defense-in-depth under the honest-but-curious model, not malicious-server integrity (see `ARCHITECTURE.md` §8). Per-block (de)serialization is a `BlockCodec` (`src/oblivlib/dependency/codec.py`) — `DefaultCodec` for plain values, `NodeCodec` (parametrized by `AVLData`/`BPlusData`) for node values — so a value-carrying scheme provides a codec instead of reimplementing path encrypt/decrypt.

## Conventions

- All public/shared methods on base classes use single-underscore (not name-mangled `__`) names so subclasses can access them — this is intentional and noted in `tree_base_oram.py`.
- Everything in `src/oblivlib/dependency/` is re-exported from `oblivlib.dependency`, so import from the package (`from oblivlib.dependency import BinaryTree, Data, ...`) rather than submodules.
- `demo/` contains runnable client/server examples over ZeroMQ sockets (`server.py` plus `oram_client.py`, `omap_client.py`, etc.) — the canonical reference for remote deployment.
- `src/oblivlib/dependency/` is kept clean: a comment earns its place only if the code can't be understood without it. Names and types speak for themselves, so docstrings that merely restate a signature are dropped; keep module docstrings and the non-obvious *why* (security, invariants, format gotchas), tightened to one line. Tests hold the same bar but lean toward keeping intent/why comments (assertions rarely reveal *why* an edge matters or what value is expected). A trivial helper already exercised by a higher-level test need not have its own dedicated test; the tests that remain order their methods to mirror the source function order — add a test next to the sibling of the source function it exercises, not at the end of the file.
