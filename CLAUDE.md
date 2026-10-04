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
- Type-check: `basedpyright` (config in `pyrightconfig.json`, `recommended` mode — the same checker Zed runs, so editor and CI agree). `recommended` is `standard` plus basedpyright's stricter checks; the dynamic-typing rules (`reportUnknown*`, `reportAny`, `reportMissingParameterType`, etc.) are explicitly disabled in the config because this library is generic over arbitrary data (`Data.key`/`value` are `Any`, values are pickled), so `Any`/`Unknown` is pervasive by design — not because of the deps, which are fully typed. See the comment block there. The whole package is type-clean. basedpyright reads `pyrightconfig.json` and must be pointed at the env where the deps are installed (via `--pythonpath` or your editor's interpreter selection); otherwise every third-party import fails to resolve and cascades into spurious errors.
- Lint/format: `ruff check` and `ruff format --check` (config under `[tool.ruff]` in `pyproject.toml`).

CI (`.github/workflows/ci.yml`) installs `.[dev]` and runs `ruff check`, `ruff format --check`, `basedpyright`, and `pytest` on push/PR to `main`.

## Architecture

### Client/server split
The central abstraction is `InteractServer` (`src/oblivlib/dependency/interact_server.py`). All ORAM/OMAP constructions talk to storage only through this interface — they never touch storage directly. Three implementations:
- `InteractLocalServer` — storage lives in the same process; used by all tests and for local development.
- `InteractRemoteServer` — client side; serializes batched queries and sends them over a socket.
- `RemoteServer` — server side; receives requests, runs them against local storage, returns results.

Construct a scheme with a `client` (an `InteractServer`), call `init_server_storage(...)` to populate storage, then issue operations. Swapping local for remote is transparent to the algorithm code.

### Query batching model
Operations do not perform I/O immediately. A construction accumulates queries on the client via `add_read_path` / `add_write_path` / `add_read_list` / `add_write_list` (each keyed by a string `label` identifying the storage; list writes are an ordered sequence of `ListWrite` / `ListPushFront` / `ListPopBack` ops), then calls `client.execute()`, which ships them as one `Request`, runs all pending **writes first, then reads**, and returns an `ExecuteResult` (`results` dict keyed by label, and `error`: the exception a failed `execute()` caught, else `None`); schemes read a result via `result.require(label)`, which re-raises that original error on a failed `execute()` and raises `MissingResultError` for a label with no result. Results and stored writes are copies (the local server pickles both directions, like the wire), so a scheme must never rely on aliasing server storage. The client-side handle tracks bandwidth (`get_bandwidth`, `reset_bandwidth`) for experiments; local and remote report identical numbers. `init_storage` raises `DuplicateLabelError` for an already-hosted label. Multiple ORAMs/OMAPs sharing one client must use distinct `name`/`label` values so their storage doesn't collide, and distinct `filename`s when file-backed (a file-backed tree truncates its file on construction).

### Storage and data types
- `init_storage` takes `dict[str, BinaryTree | list]` (`ServerStorage`) — each label maps to either a `BinaryTree` (path-ORAM tree) or a plain list; the local server keeps them in separate tree and list tables.
- `BinaryTree` (`src/oblivlib/dependency/binary_tree.py`) holds buckets of blocks. The heap-index math it and the eviction logic share (`compute_level`, `path_to_root`, `union_of_paths`, `empty_path`, `fill_data_to_path`, the O(1) closed-form `leaf_lca`) lives in `heap_index.py` as module functions. It delegates to `Storage` (`storage.py`), which owns the scheme's `BlockCodec` and fronts `_MemoryBackend` / `_FileBackend`; it never stores the encryptor. Encryption is **per bucket** (one ciphertext per bucket, via `BlockCodec.seal_bucket`/`open_bucket`); the file backend is two-phase (codec-encoded plaintext slots during fill, then a streaming `seal()`). Every scheme builds its tree through `TreeStorageBase._build_tree`. See `ARCHITECTURE.md` §3/§6.
- Core data classes and type aliases live in `src/oblivlib/dependency/types.py`: `Data` (a block: `key`, `leaf`, `value`, serialized by the shared `FieldTuplePickle` mixin it shares with `AVLData`/`BPlusData`), `PathData` (path query payload), the `ListOp` list-write ops, and the `UNSET` sentinel used to distinguish "read" from "write a value of `None`".

### ORAM layer (`src/oblivlib/oram/`)
Every scheme is constructed from a single frozen, keyword-only **config object** (defaults centralized in `src/oblivlib/dependency/config.py`), e.g. `PathOram(PathOramConfig(num_data=.., data_size=.., client=..))`; the config hierarchy mirrors the scheme hierarchy (`OramConfig` → `Da/Freecursive/Recursive/MulPath...Config`, `OmapConfig` adds `key_size`). See `ARCHITECTURE.md` §3 "Construction configs" (configs validate in `__post_init__`). `TreeBaseOram` (`tree_base_oram.py`) is the abstract base for all tree-based ORAMs; it and the OMAP `OstBaseOmap` both inherit `TreeStorageBase` (`src/oblivlib/dependency/tree_storage_base.py`), which owns the shared config accessors, leaf math, stash, and path encryption/decryption. `TreeBaseOram` adds the position map and stash eviction (`_evict_stash`, `_retrieve_data_block`). Subclasses implement `init_server_storage`, `operate_on_key`, `operate_on_key_without_eviction`, and `eviction_with_update_stash`. The key public operation is `operate_on_key(key, value=UNSET)` — always returns the current value, and writes `value` if one is provided. Concrete schemes: `PathOram`, `RecursivePathOram`, `FreecursiveOram` (insecure with `reset_method="hard"`, secure with `"prob"`), `DAOram`, `MulPathOram`, `StaticOram`.

### OMAP layer (`src/oblivlib/omap/`)
Two families:
- **Oblivious Search Tree (ODS)** maps, based on `OstBaseOmap` (`ost_base_omap.py`) — an abstract base that, like `TreeBaseOram`, inherits `TreeStorageBase`, but maintains a `root`, `local`, and `stash`. It exposes `insert`, `search` (the streaming descent that absorbed the old `fast_search`), and `delete`. Concrete: `AVLOmap`, `BPlusOmap`, and cache-optimized `AVLOmapCached` / `BPlusOmapCached`. The `distinguishable` config flag (default `False`) trades op-type hiding for speed; the cached variants are deliberately not per-op oblivious (see `ARCHITECTURE.md` §5).
- **Composed maps**: `OramOstOmap` (`oram_ost_omap.py`) is the VLDB 2025 framework — it combines *any* `TreeBaseOram` with *any* `OstBaseOmap`, hashing keys via a PRF into the ORAM. `GroupOmap` uses a group-by-hash design with an upper ORAM for metadata and a `MulPathOram` lower ORAM for key-value pairs.

### Crypto (`src/oblivlib/dependency/crypto.py`)
`Encryptor` is the abstract encryption interface (`AesGcm` is the AES-GCM implementation; `ciphertext_length` must be exact), alongside `PRP` (`FeistelPrp`) and PRF (`Blake2Prf`) primitives, and the key-hashing module functions `key_to_bytes`, `hash_data_to_leaf`, `hash_data_to_map` used by the composed maps. Constructions take an `encryptor` in their config; when `None`, data is stored in plaintext (paths skip the encrypt/decrypt step) — useful for debugging. Storage never keeps the encryptor (it is passed to `Storage.seal`), so a tree shipped to a remote server never carries the key. Buckets are encrypted as a unit (per-bucket ciphertext); GCM auth is defense-in-depth under the honest-but-curious model, not malicious-server integrity (see `ARCHITECTURE.md` §6). Per-block (de)serialization is a `BlockCodec` (`src/oblivlib/dependency/codec.py`) — `DefaultCodec` for plain values, `NodeCodec` (parametrized by `AVLData`/`BPlusData`) for node values — so a value-carrying scheme provides a codec instead of reimplementing path encrypt/decrypt.

## Conventions

- All public/shared methods on base classes use single-underscore (not name-mangled `__`) names so subclasses can access them — this is intentional and noted in `tree_base_oram.py`.
- Public names are re-exported from `oblivlib.dependency`; its `__all__` is the authoritative list (`python -c "import oblivlib.dependency as d; print(d.__all__)"`). Import those from the package (`from oblivlib.dependency import BinaryTree, Data, ...`). Internal machinery is deliberately not re-exported and is imported from its submodule: `codec`, `heap_index`, `load_bound`, `storage`, `tree_storage_base`, `flexible_binary_tree`, and the `Block`/`Bucket`/`Request` types in `types`. (`config` names are re-exported, but scheme modules also import them from `oblivlib.dependency.config`.)
- `demo/` contains runnable client/server examples over ZeroMQ sockets (`server.py` plus `oram_client.py` and `omap_client.py`) — the canonical reference for remote deployment.
- **No comments** in source or tests (tool pragmas like `# noqa` excepted). Names and types carry the meaning; any rationale the code cannot express — security arguments, invariants, format gotchas, round-count derivations — goes in `ARCHITECTURE.md`, not inline. Docstrings that restate a signature are dropped.
- A trivial helper already exercised by a higher-level test need not have its own dedicated test; tests order their methods to mirror the source function order — add a test next to the sibling of the source function it exercises, not at the end of the file.
