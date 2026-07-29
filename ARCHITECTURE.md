# DAORAM — Design Knowledge Base

This document is the design reference for the `oblivlib` library. It is written for someone (human or
agent) who needs to understand *how* the system is put together before changing it. For the
paper-to-file mapping of each algorithm, see `README.md`; for build/test commands and conventions,
see `CLAUDE.md`.

---

## 1. What this library is, and the threat model

`oblivlib` implements oblivious data structures — Oblivious RAM (ORAM), Oblivious Map (OMAP), and
oblivious graph processing — for a **client/server** deployment:

- The **client** is trusted and holds all secret state (position map, stash, encryption keys). It may
  compute non-obliviously.
- The **server** is **honest-but-curious**: it follows the protocol and only stores encrypted blocks,
  answering batched read/write requests. It must not learn the access pattern.

Because the server is honest-but-curious (not malicious), the wire format does not need to defend
against hostile responses — see §10.

---

## 2. The storage abstraction (`InteractServer`) — the central seam

`src/oblivlib/dependency/interact_server.py`. Everything that touches storage goes through this interface;
no construction reads or writes storage directly. The object is passed to every scheme as `client=`
(note the naming: the `client` argument **is** an `InteractServer` — the client-side handle to the
server, not the server itself).

Three implementations, interchangeable without changing scheme code:

| Class | Role |
|---|---|
| `InteractLocalServer` | Storage lives in-process. Used by all tests and local dev. |
| `InteractRemoteServer` | Client side: serializes batched queries and sends them over a socket. |
| `RemoteServer` | Server side: receives requests, runs them on local storage, returns results. Subclasses `InteractLocalServer`. |

### Query batching model (the key idea)

Operations do **not** perform I/O immediately. The client accumulates queries in eight label-keyed
dicts (`_read_paths`, `_read_buckets`, `_read_blocks`, `_read_lists`, and the four `_write_*`),
then a single `execute()` flushes them and returns an `ExecuteResult(success, results, error)` where
`results` is keyed by label. Semantics:

- **Writes run before reads** within one `execute()`.
- Reads are **deduplicated** (`list(set(...))`).
- List writes overload the index: `>=0` overwrite, `-1` insert-at-front, `-2` pop-last.
- `clear_queries()` runs in a `finally`, so the buffer always resets.
- Bandwidth is tracked as `len(pickle.dumps(...))` of request and response (`get_bandwidth` /
  `reset_bandwidth`) — for experiments only.

Schemes read a result with `ExecuteResult.require(label)`, **not** `result.results[label]`: `require`
raises the underlying error when `execute()` failed (`success=False`) instead of letting the caller
hit a confusing `KeyError` on the empty results dict.

Because everything is label-keyed, **multiple schemes can share one client** as long as their
`name`/`label`s differ.

### Transport

`src/oblivlib/dependency/sockets.py`: `BaseSocket` abstract, `ZMQSocket` concrete (ZeroMQ `REQ`/`REP`,
`pickle` for (de)serialization). Protocol is two message types: `("init", storage)` and
`("execute", request_dict)`; the server replies `"Done!"` or an `ExecuteResult`.

---

## 3. Data and storage layer

### The block (`Data`)

`src/oblivlib/dependency/helper.py`: `Data(key, leaf, value)` dataclass. A dummy block has all-`None`
fields; `Data.is_dummy()` / `is_real()` test for that (a real block always has a non-`None` key) and
`require_leaf()` returns the leaf or raises (used at the placement sites in `BinaryTree` instead of a
bare `assert`). `UNSET` (in `types.py`) is a sentinel distinguishing "read" (`value=UNSET`) from "write
`None`" — important: `None` is a writable value.

### Padding and sizing (only when encrypting or using files)

To hide block lengths, blocks are padded to a fixed size:

- `dump()` → `pickle.dumps((key, leaf, value))`.
- `dump_pad(length)` → `dump()` + zero padding to `length`. **No length header is stored:** pickle is
  self-delimiting (`loads` stops at the STOP opcode and ignores trailing zeros), so `load_unpad()` is
  just `pickle.loads`. A pickle always ends in the STOP byte `0x2e`, never `0x00`, so the padding
  boundary is unambiguous.
- `_dumped_data_size` = `len(dump(max key/leaf/value of size data_size))` — the exact worst-case
  pickle. **Contract:** an actual value's pickled size must not exceed this, or `dump_pad` raises.
- `_disk_size` (file mode) = `encryptor.ciphertext_length(bucket_size * _dumped_data_size)` — one
  ciphertext per **bucket** (see §8), so the AES-GCM `+28` (12-byte nonce + 16-byte tag) is paid once
  per bucket, not per block.

In pure memory + plaintext mode, no padding happens — `Data` objects are stored as-is, so values can
be arbitrary picklable objects.

### `BinaryTree` and `Storage`

`binary_tree.py`: a complete binary tree flattened into a list of buckets. Key math (all integer, no
floats — see §9):

- `level = (num_data - 1).bit_length() + 1` → leaf count `2**(level-1)` is the smallest power of two ≥ num_data.
- `size = 2**level - 1`, `start_leaf = 2**(level-1) - 1` (storage index of leaf 0).
- `get_path_indices`, `get_mul_path_dict` (root-to-leaf bucket indices), `get_cross_index`
  (deepest common ancestor of two leaves — an **O(1) closed-form** bit computation: same-depth leaves'
  1-based heap indices share a binary prefix, so the top set bit of their XOR gives the LCA, replacing
  the old O(depth) parent-walk in the eviction inner loop), `fill_data_to_path` (eviction placement).

`storage.py`: a thin `Storage` facade over two interchangeable backends — `_MemoryBackend` (list of
lists of `Data`) and `_FileBackend` (a pre-allocated fixed-size byte file). **Encryption is per
bucket** (see §8). The file backend is **two-phase**: it stores plaintext block-slots while the tree is
being filled, then `encrypt()` seals each row into one ciphertext blob in a **streaming pass** (one
bucket in memory at a time, so it scales to trees larger than RAM — never materializes the whole tree).
A `_sealed` flag flips its read/write between block-slot and blob layouts. The four modes
(memory/file × plaintext/encrypted) are all internally consistent and tested
(`tests/dependency/test_storage.py`).

### Construction configs (`config.py`)

Every ORAM/OMAP scheme is constructed from a **frozen, keyword-only config object** rather than a
dozen positional params. The config hierarchy mirrors the scheme hierarchy and **centralizes
defaults** — one place per scheme, inherited and overridden:

- `OramConfig` (the shared core: `num_data, data_size, client, name, filename, bucket_size,
  stash_scale, encryptor`) → `PathOramConfig`, `StaticOramConfig`, `MulPathOramConfig`,
  `RecursiveOramConfig`, and `CounterOramConfig` → `DaOramConfig` / `FreecursiveOramConfig`.
- `OmapConfig(OramConfig)` adds `key_size` → `AvlOmapConfig`, `BPlusOmapConfig` (`order`), the cached
  variants, and `GroupOmapConfig`.

Configs **validate in `__post_init__`** (frozen, so they raise rather than mutate): shared numeric
fields in `OramConfig` (`num_data`/`data_size`/`bucket_size`/`stash_scale ≥ 1`) and scheme-specific
ones in the subclass (`reset_prob ∈ (0,1]`, `order ≥ 3`, …); `reset_method` is a `Literal["prob",
"hard"]`. Subclass `__post_init__`s chain via `super()` (dataclasses don't chain automatically), so
misuse fails locally at construction instead of deep inside a scheme.

Construction: `PathOram(PathOramConfig(num_data=…, data_size=…, client=…))`. The frozen config is the
**single source of truth**: the shared `TreeStorageBase` (§4) exposes each construction parameter
as a **read-only private property reading `self._config`** (`self._num_data`, `self._client`, …), so
method bodies are unchanged and no state is duplicated. (`client` is internal-only — nothing external
reads it — so it's `self._client`, with no separate public `client` property.) *Derived* values (`_level`, `_leaf_range`,
`_disk_size`, …) are computed once in `__init__` (not properties — they're hot); *scheme-specific*
config fields get their own properties in that scheme's file. Object params that aren't config — a
`StaticOram`'s `prf`, a `GroupOmap`'s `upper_oram` — are separate constructor args. Derive a variant
with `dataclasses.replace(cfg, num_data=…)`. (The package requires Python ≥ 3.12 — PEP 695 generics
in `TreeStorageBase` — so `kw_only` configs are always available.)

**Internal recursion plumbing stays out of the config.** `is_pos_map`, `last_oram_data`,
`last_oram_level` are not user knobs, so they are keyword-only internal `__init__` args (`_is_pos_map`,
…). A recursive scheme builds each position-map child with `replace(self._config, num_data=…,
name=f"{self._name}_pos_map_{i}")` + those internals — the child **shares the parent's client** (the
client is label-keyed, so multiple orams share one) and owns that distinct `name`, which is exactly
the label its storage lives under. Each child then drives its own per-level I/O
(`_access_pos_map_level`, reading/writing `self._name` on the shared client); the parent's recursive
loop only samples leaves and threads the leaf from one level to the next.

The loose `dict` type hints are now aliases in `types.py`: `PosMap = Dict[int, int]` (key→leaf),
`DataMap = Dict[int, Any]` (init key→value).

---

## 4. ORAM layer (`src/oblivlib/oram/`)

### `TreeStorageBase` — the shared tree-storage core

`src/oblivlib/dependency/tree_storage_base.py`. `TreeBaseOram` (ORAM) and `OstBaseOmap` (ODS OMAP,
§5) both lay a complete binary tree over `InteractServer` storage and shared an identical construction
core; that core now lives in one abstract base, `TreeStorageBase(Generic[ConfigT])`, which both
inherit. It owns the frozen config and its read-only field accessors (`_name`, `_num_data`, `_client`,
…), the integer level/leaf-range/stash-size math, the padded-block/disk sizing
(`_dumped_data_size`/`_disk_size`), `_get_new_leaf`, the stash, and **path encrypt/decrypt** (driven by
a `BlockCodec`, §8). `TreeBaseOram` adds the position map + eviction; `OstBaseOmap` adds the
root/local + tree traversal. It lives in `dependency/` so both higher layers depend *downward* on it,
never sideways on each other. (`ConfigT` is bound to `OramConfig`; `OmapConfig` is an `OramConfig`, so
the ODS parametrizes the same base.)

### `TreeBaseOram` (abstract base)

A `TreeStorageBase` (above) constructed from an `OramConfig` (§3, "Construction configs"). Adds the
**position map** (`{key: leaf}`) and eviction on top of the shared core. Subclasses implement
`init_server_storage`, `operate_on_key`, `operate_on_key_without_eviction`, `eviction_with_update_stash`.

**The access protocol** (`operate_on_key(key, value=UNSET)` — always returns the current value,
writes `value` if provided):

1. Look up `key`'s leaf in the position map; assign it a fresh random leaf.
2. **Read** the path to the old leaf (`add_read_path` + `execute`).
3. `_retrieve_data_block`: pull all real blocks on the path into the stash, find `key`, read it,
   optionally overwrite, remap it to the new leaf.
4. `_evict_stash`: push stash blocks back down the path as deep as legal.
5. **Write** the evicted path back (`add_write_path` + `execute`).

So each access is **two round trips** (read, then write). The `*_without_eviction` +
`eviction_with_update_stash` pair lets callers defer the write-back for batching.

### Eviction (`_evict_stash` / `fill_data_to_path`) — block-major, and it's optimal

For each stash block, it is placed at the **deepest legal bucket** (the deepest node on both the
block's assigned path and the eviction path, via `get_cross_index`), bubbling up toward the root if
full; blocks that don't fit stay in the stash. This "block-major" greedy was benchmarked against the
textbook "bucket-major" sweep: they produce **identical overflow** (both are maximal placements on a
laminar matroid), and block-major is **~15% faster** in the realistic small-stash regime. So the
current implementation is the right one — do not switch to bucket-major.

### Concrete schemes

| Scheme | Idea |
|---|---|
| `PathOram` | Textbook Path ORAM: random leaf reassignment each access. |
| `MulPathOram` | Path ORAM that reads/evicts **several paths per batch** (`operate_on_keys`, `operate_on_keys_without_eviction`, `eviction_for_mul_keys`); larger stash via `stash_scale_multiplier`. |
| `StaticOram` | Leaf positions are **fixed** by `PRF(key)` — no remapping on access. |
| `RecursivePathOram` | The position map is too big for the client, so it is **compressed into a chain of smaller position-map orams** (`compression_ratio` leaves per block); only a small top map (≤ `on_chip_mem`) stays on the client. |
| `DAOram`, `FreecursiveOram` | Leaves are derived as `PRF(key ‖ GC ‖ IC)`. The position maps store **counters** per key — a group count (GC) plus per-key individual counts (IC). Accessing a key bumps its IC and recomputes its leaf; when counters near overflow the block **resets** (bump GC, zero ICs) and all affected leaves are recomputed. `FreecursiveOram` triggers reset probabilistically (`reset_method="prob"`, `reset_prob ≈ 1/num_ic`) or on overflow (`"hard"`); `DAOram` de-amortizes the reset work across accesses (a reset path is carried alongside each normal access). |

`DAOram`/`FreecursiveOram`/`RecursivePathOram` are the **recursive-position-map** schemes: their
internal position-map orams are constructed with `is_pos_map=True` (no client) and recursively
smaller `num_data`.

The de-amortized counter/reset state is threaded through **named `NamedTuple` records** (`ResetLeaf`/
`ProcessedData` in `da_oram.py`; `ResetEntry`/`ResetChunk`/`ProcessedData`/`UnpackedReset` in
`freecursive_oram.py`) rather than positional tuples — fields are honestly typed, so DA's old
`cast(PROCESSED_DATA, …)` (which lied about the runtime `None`s) is gone, replaced by a narrowing
assert after the key-found guard. (NamedTuple, not dataclass, so the many positional-unpack call sites
are untouched; the `ResetLeaf.index` field is named `offset` because a NamedTuple field can't shadow
`tuple.index`.)

---

## 5. OMAP layer (`src/oblivlib/omap/`)

Two families:

- **Oblivious Search Trees (ODS):** `OstBaseOmap` base (a `TreeStorageBase`, §4) with `AVLOmap`,
  `BPlusOmap`, and cache-optimized `AVLOmapCached` / `BPlusOmapCached`. Constructed from `OmapConfig`
  subclasses (§3). Maintain a `root`, `local`, and `stash`; expose `insert`, `search` (the streaming
  descent that absorbed the old `fast_search` — padded to the op budget, so still oblivious by
  default), and `delete`.
- **Composed maps:** `OramOstOmap` (the VLDB 2025 framework — combine *any* `TreeBaseOram` with *any*
  `OstBaseOmap`, hashing keys into the ORAM via a PRF) and `GroupOmap` (group-by-hash over an
  upper metadata ORAM plus a `MulPathOram` lower ORAM).

**Obliviousness model (per-op access pattern).** Each logical op must induce a fixed-shape access
pattern. The base centralizes this: `_short_circuit_read` handles the dummy (`key is None`) / empty-tree
case for every read-like op, and each op pads to a budget from a per-scheme worst-case round map
(`_op_round_bounds`, derived from the tree height `h`). The `distinguishable` config flag (default
`False` = fully oblivious) pads insert/search/delete all to `max(bounds)` so the op *type* is hidden;
`True` lets each op use its own smaller bound. So `op(missing)` == `op(hit)` == `op(None)` in round
count. **The cached variants (`AVLOmapCached`/`BPlusOmapCached`) are deliberately NOT per-op
oblivious** — they are the optimized/amortized variants meant for use *inside* a larger oblivious
composition; the test suite encodes this with per-variant `per_op_oblivious`/`delete_oblivious` flags
(see §11). Shared node bookkeeping lives in `LocalNodesBase`; AVL/B+ `LocalNodes` extend it.

---

## 6. Graph layer (`src/oblivlib/graph/`)

`GraphOS` (VLDB 2024) and `Grove` (delayed-duplications) build on `MulPathOram` + `AVLOmapCached`,
using `src/oblivlib/dependency/graph.py` for the graph representation.

---

## 7. SORAM (`src/oblivlib/soram/`)

A weaker-threat-model ORAM that hides access patterns only over windows of `c` consecutive
operations — more efficient than full ORAM. Tests are excluded from the default `pytest` run.

---

## 8. Crypto primitives (`src/oblivlib/dependency/crypto.py`)

- `Encryptor` (abstract) → `AesGcm` (AES-GCM, returns `nonce ‖ ciphertext ‖ tag`).
- `PseudoRandomFunction` (abstract) → `Blake2Prf` (keyed BLAKE2b; `digest`, `digest_mod_n`).
- `PseudoRandomPermutation` (abstract) → `FeistelPrp` (4-round Feistel, SHA-256 round function with a
  key-seeded hash cached and `.copy()`d per round, cycle-walking for non-power-of-2 domains).

**Encryption granularity is per bucket.** Both the init seal (`Storage.encrypt`) and per-operation
path encryption (`_encrypt_path_data`) build *one ciphertext per bucket* via the shared
`Helper.encrypt_bucket` / `decrypt_bucket`: concatenate the `bucket_size` fixed-width blocks (real +
dummies) and encrypt once. This means one `nonce+tag` per bucket instead of per block (less bandwidth,
`bucket_size`× fewer crypto calls). GCM's authentication is **defense-in-depth** under the
honest-but-curious model (§1), not required for security — and note that per-bucket GCM is **not**
malicious-server integrity: it has no replay/freshness protection (that would need a Merkle/version
tree over the access path). The 96-bit random nonce is standard but caps safe use at ~2³² encryptions
per key; `AESGCMSIV` would lift that ceiling if ever needed at scale.

**Per-block serialization is a `BlockCodec`** (`src/oblivlib/dependency/codec.py`). The bucket seal above
packs fixed-width per-block payloads; turning one `Data` into that payload (and back) is the codec's
job, so `TreeStorageBase._encrypt/_decrypt_path_data` (§4) are written once and driven by
`self._codec`. `DefaultCodec` stores the value verbatim (ORAM / plain ODS); `NodeCodec` (parametrized
by the node class, `AVLData` / `BPlusData`) stores the node's value as its own pickle bytes and
rebuilds it on load. A codec
**builds the payload without mutating the live block** — the old AVL/B+ path did an in-place
`data.value = value.dump()`, a latent hazard now gone. A new value-carrying scheme provides a codec
instead of copying the encrypt/decrypt methods (AVL/B+ override only `_codec`).

When a scheme's `encryptor` is `None`, data is stored in plaintext (the encrypt/decrypt steps are
skipped) — useful for debugging and used by the tests for speed.

---

## 9. Invariants and contracts

- **Path ORAM invariant:** a block assigned to leaf `x` is always on the root→`x` path or in the stash.
- **Level math is integer:** `level = (num_data-1).bit_length() + 1` in *both* `BinaryTree`
  (`compute_level`) and `TreeStorageBase.__init__`; the two must stay in agreement.
- **Recursive-pos-map schemes require `num_data > on-chip size`** (`on_chip_mem` for DA/Recursive,
  `on_chip_size` for Freecursive). Smaller values leave no position-map levels; the constructor now
  raises a clear `ValueError` instead of crashing deep inside.
- **Value size:** in encrypted/file mode, a value's pickled size must be ≤ `data_size`.
- **Per-bucket encryption is self-consistent:** the init seal (`Storage.encrypt`) and the
  per-operation `_encrypt_path_data` must produce the *same* per-bucket blob layout — same
  `block_size` and per-block prep — or `decrypt_bucket` splits at the wrong boundaries. `block_size`
  is now `codec.block_size` (§8): `_dumped_data_size` for the `DefaultCodec` (ORAM/ODS),
  `_max_block_size` for the `NodeCodec`s, and is also what each scheme passes as the Storage
  `data_size`. The init seal does not call the codec, but the codec reproduces the seal's per-block
  bytes exactly — keep them in step.
- **Shared client:** multiple schemes on one client must use distinct `name`s.
- **`UNSET` vs `None`:** `operate_on_key(key)` reads; `operate_on_key(key, None)` writes `None`. The
  return value is always the value *before* any write.

---

## 10. Known issues / gotchas

- **`FlexibleBinaryTree` (`src/oblivlib/dependency/flexible_binary_tree.py`) is revived but still orphaned.**
  It was non-functional (couldn't construct against the current `Storage`, old float math, dead
  module-level pickle code). It is now wired to the current `Storage` API (mirroring `BinaryTree`:
  `encryption: bool` + `data_size`/`disk_size`; the caller drives encryption), uses integer math, has
  the dead code removed, and has a characterization suite (`tests/dependency/test_flexible_binary_tree.py`,
  36 tests). It is still **not imported anywhere** (kept for future soram use), and its
  `scale_up`/`scale_down` semantics are *characterized, not specified* — when soram adopts it, validate
  the scaling against the real requirements. `get_cross_index` here takes raw storage indices (not leaf
  labels) and assumes same-depth inputs (callers align via `adjust_to_same_level`).
- **`src/oblivlib/graph/` (`GraphOS`, `Grove`) is broken against the current API and untested.** Both call
  ORAM/OMAP methods that no longer exist — `MulPathOram.retrieve_path`/`evict_path`/
  `process_path_to_stash`/`prepare_evict_path`, `InteractServer.read_mul_query`/`write_mul_query`,
  `AVLOmapCached.search_with_meta` — because the batch ORAM API was reshaped to
  `operate_on_keys`/`eviction_for_mul_keys` and graph was never ported. It imports fine but raises on
  first real use; there are no graph tests or callers. Like `soram`, it is **excluded from
  `pyrightconfig.json`** pending a deliberate port to the current API (do it with tests).
- **Type checking.** The package is **pyright-clean** (basedpyright `recommended` mode; see `pyrightconfig.json`) across
  `dependency`/`oram`/`omap`, with **no `# type: ignore`s**. `soram` and `graph` are excluded as
  broken-pending-port (above). The ODS base `OstBaseOmap` is **generic over its local-node
  container** (`LocalT` bound to `LocalNodesBase`), so AVL/B+ supply `LocalNodes` as a type argument
  (`OstBaseOmap[AvlOmapConfig, LocalNodes]`) instead of narrowing a base `_local` attribute —
  which is what used to require a `reportIncompatibleVariableOverride` suppression.
- **Wire format is `pickle`.** Safe under the honest-but-curious model (§1), but it would be a remote
  code-execution risk against a *malicious* server, and the bandwidth numbers include pickle framing
  overhead rather than raw ciphertext volume.
- **Two round trips per access.** `operate_on_key` does a read `execute()` then a write `execute()`.
  The deferred-eviction primitives could amortize this to one round trip per access (piggyback access
  N's write-back on access N+1's read), but the default path does not.

---

## 11. Testing architecture (`tests/`)

- `tests/conftest.py`: shared fixtures (`num_data` — default `2**12`, override a single run with the
  `NUM_DATA` env var — `client`, `encryptor`, `test_file`) and the soram exclusion. No rootdir
  conftest: using an env var instead of a custom `--option` avoids the `pytest_addoption`-must-load-
  before-arg-parsing constraint that would force a top-level conftest.
- `tests/dependency/`: `conftest.py` exposes `make_search_tree` — a parametrized factory (AVL / B+)
  so one behavioral suite (`test_search_tree_common.py`) covers both trees (int/str keys, missing-key
  contract, delete + stress). Scheme-specific *structural* assertions stay in `test_avl_tree.py` /
  `test_bplus_tree.py`, each with a full tree-invariant validator (BST + balance + heights for AVL;
  sorted keys, child-count, uniform leaf depth for B+). `test_storage.py` is parametrized over the
  four backends and characterizes the per-bucket encrypt/decrypt round-trip, full-bucket/multi-row,
  construction `ValueError`s, the padding boundary, and disk reopen. `test_binary_tree.py` checks the
  index math (incl. exactness at deep levels and the closed-form LCA); `test_crypto.py` /
  `test_helper.py` cover the primitives and boundary cases. Within each per-module file the test
  methods are ordered to mirror the function order of the `dependency/` source they cover, so a source
  module and its test read top-to-bottom in lock-step. Coverage is not 1:1, though: a trivial helper
  already exercised through a higher-level test (e.g. `get_parent_index` via `get_path_indices`) carries
  no dedicated test of its own.
- `tests/oram/conftest.py`: the unification machinery.
  - `make_oram` — a **parametrized factory fixture**; any test taking it runs once per ORAM type.
    Each param is a factory taking common kwargs (`num_data, data_size, client, filename, encryptor`).
    `FreecursiveOram` appears twice (prob/hard reset).
  - `storage_kwargs` — parametrized over the four backends (memory/file × plaintext/encrypted).
- `tests/oram/test_oram_common.py`: the shared single-key behavioral suite. `test_round_trip` crosses
  `make_oram × storage_kwargs` (every ORAM × every backend) in one method.
- `tests/oram/test_mul_path_oram.py`: `MulPathOram`'s **batch-only** API (`operate_on_keys`, etc.).
  It is separate not because MulPathOram is excluded from the common suite (it isn't — it's in
  `ORAM_SPECS`) but because these methods exist on no other scheme.
- `tests/oram/test_oram_edge_cases.py`: non-power-of-two sizes, the level/leaf-range invariant, the
  `num_data > on-chip` guard, the write-returns-old-value / `None`-is-writable contract, and
  arbitrary (non-int) value types.
- `tests/oram/test_oram_integration.py`: the remote client/server protocol (`InteractRemoteServer` +
  `RemoteServer`, exercised through an in-process loopback socket that pickles like the wire) and
  multiple ORAMs sharing one client under distinct names.

**To add a new ORAM:** add one line to `ORAM_SPECS` and the whole common suite covers it.
**To add a new shared behavior:** add a method to `TestOramCommon` and every ORAM runs it.

`tests/omap/` now has the dependency/oram-style unification: `conftest.py` exposes the `omap_spec`
factory over `OMAP_SPECS` (AVL / B+ × cached / non-cached, each recording its guarantees:
`per_op_oblivious`, `delete_oblivious`, …), and `test_omap_common.py` runs one shared suite over it —
`TestOmapBehavior` (model oracle, invariants, edge sizes, string keys, encryption/file backends,
init-with-data, delete oracle) and `TestOmapObliviousness` (per-op access-pattern uniformity). The
obliviousness tests draw from **filtered fixtures** (`oblivious_omap_spec` / `delete_oblivious_omap_spec`)
so they are *collected* only for the variants that claim the property — the cached variants are excluded
by design rather than skipped at runtime (so the suite has no standing skips). `test_codec.py` checks
the `BlockCodec` round-trip and that `dump_block` does not mutate the live block. Scheme-specific tests
stay in `test_avl_omap.py` etc.; `test_group_omap.py` covers `GroupOmap`.

**To add a new OMAP:** add one line to `OMAP_SPECS` with its capability flags and the whole common
suite covers it (and only the obliviousness tests its flags opt into).

**Remaining coverage gaps:** the live `ZMQSocket` transport over a real TCP connection (the
client/server *logic* is covered via the loopback socket, but ZMQ itself is not); the
`get_bandwidth`/`reset_bandwidth` counters; and degenerate inputs like an empty `operate_on_keys({})`
batch or a forced stash-overflow `MemoryError`.
