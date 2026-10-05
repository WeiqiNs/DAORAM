# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`oblivlib` is a Python library implementing classic oblivious algorithms — Oblivious RAM (ORAM) and Oblivious Map (OMAP). It targets a client/server deployment where the client is trusted and may compute non-obliviously, while the server only stores encrypted data and answers batched read/write queries. The library backs the maintainers' publications (notably "Towards Practical Oblivious Map", VLDB 2025). See `README.md` for the paper-to-file mapping of each algorithm.

**For the full design knowledge base — data flow, the storage/ORAM/OMAP layers, the position-map and counter schemes, eviction, invariants/contracts, the test architecture, and known issues — read `ARCHITECTURE.md`.** This file is a quick reference; `ARCHITECTURE.md` is the deep one.

## Commands

- Run all tests: `pytest`. All of `tests/dependency`, `tests/oram`, and `tests/omap` collect and pass.
- Run a single test file: `pytest tests/oram/test_oram_common.py`
- Run a single test: `pytest "tests/oram/test_oram_common.py::TestOramCommon::test_round_trip"`
- Control dataset size in tests: the `num_data` fixture defaults to `2**12` (`DEFAULT_NUM_DATA` in `tests/conftest.py`); override a single run via the `NUM_DATA` env var — smaller for a fast local loop, larger to stress — e.g. `NUM_DATA=256 pytest tests/oram/test_oram_common.py`.
- Install for development: `pip install -e ".[dev]"` (runtime deps plus `pytest`, `basedpyright`, and `ruff`). All project metadata and dependencies live in `pyproject.toml` — runtime under `[project.dependencies]`, dev tooling under the `dev` optional-dependencies extra.
- Type-check: `basedpyright` (config in `pyrightconfig.json`, `recommended` mode — the same checker Zed runs, so editor and CI agree). `recommended` is `standard` plus basedpyright's stricter checks; the dynamic-typing rules (`reportUnknown*`, `reportAny`, `reportMissingParameterType`, etc.) are explicitly disabled in the config because this library is generic over arbitrary data (`Data.key`/`value` are `Any`), so `Any`/`Unknown` is pervasive by design — not because of the deps, which are fully typed. See the comment block there. The whole package is type-clean. basedpyright reads `pyrightconfig.json` and must be pointed at the env where the deps are installed (via `--pythonpath` or your editor's interpreter selection); otherwise every third-party import fails to resolve and cascades into spurious errors.
- Lint/format: `ruff check` and `ruff format --check` (config under `[tool.ruff]` in `pyproject.toml`).

CI (`.github/workflows/ci.yml`) installs `.[dev]` and runs `ruff check`, `ruff format --check`, `basedpyright`, and `pytest` on push/PR to `main`.

## Architecture

### Client/server split
All ORAM/OMAP constructions talk to storage only through a `Client` (`src/oblivlib/dependency/client.py`), passed in their config as `client=` — they never touch storage directly. The layers below it:
- `StorageServer` (`storage_server.py`) is the storage engine: it answers the protocol messages of `protocol.py` and has no transport of its own. Trees live in fixed-row stores (memory, or files under its `storage_dir`) or, for plaintext debugging, variable-row memory stores (`server_stores.py`).
- A `Backend` carries messages: `LocalBackend` calls the server in process (rows are immutable bytes, so nothing is copied); `TransportBackend` encodes them for a byte-level `Transport` (`transport.py`: `ZmqTransport`, `LoopbackTransport`, `SimulatedNetwork`; the server side is `serve(ZmqListener(...), server)`).
- `Client.local(storage_dir=None)` and `Client.connect(endpoint)` build the two usual stacks. Swapping local for remote is transparent to the algorithm code.

Construct a scheme with a `client`, call `init_server_storage(...)` (the client builds the tree locally and hosts it), then issue operations. `ARCHITECTURE.md` §2 is the wire spec.

### Query batching model
Operations do not perform I/O immediately. A construction stages queries on the client via `add_read_path` / `add_write_path` / `add_read_list` / `add_write_list` (each keyed by a string `label`), then calls `client.execute()`, which sends one batch whose **writes apply before its reads** and returns an `ExecuteResult`; schemes read a result via `result.require(label)`. A failed batch raises the server's error, mapped to its class in `errors.py`. Path rows travel as `PathRows` (`{heap_index: row}`, `b""` = absent node) and the scheme's `PathCipher` turns them into buckets. By default the client **defers writes**: an `execute()` that stages only writes sends nothing, and its writes ride on the next batch — the access sequence is unchanged and the round count halves; `flush()` and `close()` send them. `client.metrics` reports rounds, payload bytes, and wire bytes. One `Client` per thread. ORAMs/OMAPs sharing one client must use distinct `name`s (their storage labels).

### Storage and data types
- The data contract is bytes: ORAM keys are int addresses in `[0, num_data)`, OMAP keys are `bytes` of at most `key_size`, values are `bytes` of at most `data_size`. `contract.py` checks it at every public entry point and raises `ContractError` before any state changes. The library never pickles.
- Core data classes and type aliases live in `src/oblivlib/dependency/types.py`: `Data` (a block: `key`, `leaf`, `value`, whose field order — the `FieldTuple` mixin it shares with `AVLData`/`BPlusData` — is the stored format), `PathData`/`PathRows`, the `ListOp` list-write ops, and the `UNSET` sentinel that distinguishes "read" from "write" in `operate_on_key`.
- A `BlockCodec` (`codec.py`) turns a block into the msgpack-ready `[key, leaf, value]` — `DefaultCodec` for plain values, `NodeCodec` for AVL/B+ node values. A `PathCipher` (`path_cipher.py`) packs each bucket as a msgpack array of its blocks and, when encrypted, seals it into one fixed-length row (one ciphertext per bucket). See `ARCHITECTURE.md` §3/§6.
- `build_tree` (`tree_builder.py`) builds the initial tree on the client, in memory or in an encrypted `build_file`; `Client.host_tree` hands it to the server (adopt, attach, or stream). Every scheme builds through `TreeStorageBase._build_tree`. Heap-index math (`compute_level`, `path_indices`, `leaf_lca`, `fill_data_to_path`, ...) lives in `heap_index.py`.

### ORAM layer (`src/oblivlib/oram/`)
Every scheme is constructed from a single frozen, keyword-only **config object** (defaults centralized in `src/oblivlib/dependency/config.py`), e.g. `PathOram(PathOramConfig(num_data=.., data_size=.., client=..))`; the config hierarchy mirrors the scheme hierarchy (`OramConfig` → `Da/Freecursive/Recursive/MulPath...Config`, `OmapConfig` adds `key_size`). See `ARCHITECTURE.md` §3 "Construction configs" (configs validate in `__post_init__`). `TreeBaseOram` (`tree_base_oram.py`) is the abstract base for all tree-based ORAMs; it and the OMAP `OstBaseOmap` both inherit `TreeStorageBase` (`src/oblivlib/dependency/tree_storage_base.py`), which owns the shared config accessors, leaf math, stash, and path encryption/decryption. `TreeBaseOram` adds the position map and stash eviction (`_evict_stash`, `_retrieve_data_block`). Its public `operate_on_key`, `operate_on_key_without_eviction`, and `eviction_with_update_stash` check the data contract and then call the underscore versions each subclass implements, alongside `init_server_storage(data=...)` (a mapping or a one-shot stream of `(key, value)` pairs). The key public operation is `operate_on_key(key, value=UNSET)` — always returns the current value (`b""` until written), and writes `value` if one is provided. Concrete schemes: `PathOram`, `RecursivePathOram`, `FreecursiveOram` (insecure with `reset_method="hard"`, secure with `"prob"`), `DAOram`, `MulPathOram`, `StaticOram`.

### OMAP layer (`src/oblivlib/omap/`)
Two families:
- **Oblivious Search Tree (ODS)** maps, based on `OstBaseOmap` (`ost_base_omap.py`) — an abstract base that, like `TreeBaseOram`, inherits `TreeStorageBase`, but maintains a `root`, `local`, and `stash`. It exposes `insert`, `search` (the streaming descent that absorbed the old `fast_search`), and `delete`, which check the contract and call each scheme's `_insert` / `_search` / `_delete`. Concrete: `AVLOmap`, `BPlusOmap`, and cache-optimized `AVLOmapCached` / `BPlusOmapCached`. The `distinguishable` config flag (default `False`) trades op-type hiding for speed; the cached variants are deliberately not per-op oblivious (see `ARCHITECTURE.md` §5).
- **Composed maps**: `OramOstOmap` (`oram_ost_omap.py`) is the VLDB 2025 framework — it combines *any* `TreeBaseOram` with *any* `OstBaseOmap`, hashing keys via a PRF into the ORAM, which stores each slot's msgpack-encoded root (size its `data_size` with `OramOstOmap.oram_data_size`). `GroupOmap` uses a group-by-hash design with an upper ORAM for metadata and a `MulPathOram` lower ORAM whose blocks are named by the user's key.

### Crypto (`src/oblivlib/dependency/crypto.py`)
`Encryptor` is the abstract encryption interface (`AesGcm` is the AES-GCM implementation; `ciphertext_length` must be exact), alongside `PRP` (`FeistelPrp`) and PRF (`Blake2Prf`) primitives, and the key-hashing module functions `hash_data_to_leaf`, `hash_data_to_map` used by the composed maps. Constructions take an `encryptor` in their config; when `None`, buckets are stored as plaintext msgpack rows of varying length (memory servers only) — useful for debugging. Only the client holds the encryptor; the server only ever sees sealed rows. Buckets are encrypted as a unit (per-bucket ciphertext, fixed row length); GCM auth is defense-in-depth under the honest-but-curious model, not malicious-server integrity (see `ARCHITECTURE.md` §6). A value-carrying scheme provides a `BlockCodec` instead of reimplementing path encryption.

## Conventions

- All public/shared methods on base classes use single-underscore (not name-mangled `__`) names so subclasses can access them — this is intentional and noted in `tree_base_oram.py`.
- Public names are re-exported from `oblivlib.dependency`; its `__all__` is the authoritative list (`python -c "import oblivlib.dependency as d; print(d.__all__)"`). Import those from the package (`from oblivlib.dependency import Client, Data, ...`). Internal machinery is deliberately not re-exported and is imported from its submodule: `protocol` (the wire messages), `codec`, `contract`, `heap_index`, `load_bound`, `path_cipher`, `server_stores`, `tree_storage_base`, `flexible_binary_tree`, and the `Bucket` type in `types`. (`config` names are re-exported, but scheme modules also import them from `oblivlib.dependency.config`.)
- `demo/` contains runnable client/server examples over ZeroMQ (`server.py` plus `oram_client.py` and `omap_client.py`) — the canonical reference for remote deployment.
- **No comments** in source or tests (tool pragmas like `# noqa` excepted). Names and types carry the meaning; any rationale the code cannot express — security arguments, invariants, format gotchas, round-count derivations — goes in `ARCHITECTURE.md`, not inline. Docstrings that restate a signature are dropped.
- A trivial helper already exercised by a higher-level test need not have its own dedicated test; tests order their methods to mirror the source function order — add a test next to the sibling of the source function it exercises, not at the end of the file.
