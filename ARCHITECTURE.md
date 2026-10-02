# oblivlib — Design Knowledge Base

How the library is put together, for anyone (human or agent) about to change it. The source carries no
comments by convention: any rationale the code cannot express lives here. Paper-to-file mapping is in
`README.md`; commands and conventions are in `CLAUDE.md`.

---

## 1. Scope and threat model

`oblivlib` implements oblivious data structures (ORAM, OMAP, oblivious graph processing) for a
client/server deployment:

- The **client** is trusted, holds all secret state (position map, stash, keys), and may compute
  non-obliviously.
- The **server** is **honest-but-curious**: it follows the protocol, stores only encrypted blocks, and
  answers batched read/write requests. It must not learn the access pattern.

Because the server is not malicious, the wire format does not defend against hostile responses (§10).

---

## 2. The storage seam (`InteractServer`)

`src/oblivlib/dependency/interact_server.py`. Every storage access goes through this interface. Schemes
receive it as `client=`: it is the client-side handle *to* the server, not the server.

| Class | Role |
|---|---|
| `InteractLocalServer` | Storage in-process. Used by tests and local runs. |
| `InteractRemoteServer` | Client side: ships batched queries over a socket. |
| `RemoteServer` | Server side: runs received queries on local storage. Subclasses `InteractLocalServer`. |

**Query batching.** No call does I/O immediately. Queries accumulate in eight label-keyed buffers
(`_read_{paths,buckets,blocks,lists}`, `_write_*`); `execute()` flushes them and returns
`ExecuteResult(success, results, error)` keyed by label.

- Writes run before reads within one `execute()`; a later write to the same slot overwrites an earlier one.
- Reads are deduplicated. `add_read_list(label, None)` reads the whole list.
- List-write indices are sentinels: `>= 0` overwrites, `-1` inserts at the front, `-2` pops the last.
- The buffers are cleared in a `finally` on both local and remote clients, so a failed `execute()`
  never leaks queries into the next one.
- Bandwidth (`get_bandwidth`/`reset_bandwidth`) is `len(pickle.dumps(...))` of request and response,
  for experiments only.
- Read results through `ExecuteResult.require(label)`: on a failed `execute()` it raises the original
  error instead of a masking `KeyError` on the empty results dict.

Storage is `dict[label, BinaryTree | list]`. Tree labels and list labels are kept disjoint by callers,
which is what lets `_get_tree`/`_get_list` cast. Schemes sharing one client must use distinct names.

**Transport.** `sockets.py`: `ZMQSocket` (ZeroMQ `REQ`/`REP`, pickle framing). Two messages:
`("init", storage)` → `"Done!"`, and `("execute", request_dict)` → `ExecuteResult`.

---

## 3. Data and storage layer

### Blocks

`Data(key, leaf, value)` (`helper.py`). A dummy block has `key is None`. `UNSET` (`types.py`)
distinguishes "read" from "write `None`": `None` is a writable value.

### Padding and sizing (encryption or file storage only)

- `dump_pad(length)` = `pickle.dumps((key, leaf, value))` + zero padding. Fields are pickled shallowly
  (not via `dataclasses.astuple`, which would flatten a dataclass value into a tuple). There is no length header:
  pickle is self-delimiting and always ends in STOP (`0x2e`, never `0x00`), so `load_unpad` is plain
  `pickle.loads`.
- `_dumped_data_size` is the exact worst-case pickle for keys/leaves/values at the configured sizes. A
  value whose pickle exceeds it makes `dump_pad` raise.
- `_disk_size` (file mode) derives from the scheme's codec: `ciphertext_length(bucket_size * block_size)`
  when encrypted (one ciphertext per bucket), else `block_size` (one slot).

Memory + plaintext mode stores `Data` objects as-is, so values may be any picklable object.

### `BinaryTree` and `Storage`

`BinaryTree` flattens a complete binary tree into a heap-indexed bucket list, with all math in integers:
`level = (num_data - 1).bit_length() + 1`, `size = 2**level - 1`, and `start_leaf = 2**(level-1) - 1`
(the internal-node count, i.e. the storage index of leaf 0).

- `get_cross_index` is O(1): same-depth leaves' 1-based heap indices share a binary prefix, so shifting
  past the top set bit of their XOR yields the lowest common ancestor.
- `fill_data_to_path` places a block at the deepest bucket on both its own path and the target path
  set, bubbling up toward the root if full.
- Storage hands back bucket copies in file mode, so every in-place edit (`write_block`,
  `fill_data_to_storage_leaf`) reads the bucket, mutates it, and writes it back explicitly.
- `BinaryTree` allocates all `2**level - 1` buckets eagerly. Never construct one at large `num_data`
  just to read its level; use `BinaryTree.compute_level`.

`Storage` fronts two backends: `_MemoryBackend` (a list of buckets) and `_FileBackend` (a pre-allocated
fixed-row file). The file backend is two-phase. While the tree is filled, rows hold `bucket_size`
fixed-width plaintext block slots. `encrypt()` then streams one row at a time, sealing each into a
single blob and flipping `_sealed`, so trees larger than RAM never materialize. Rows are sized for the
sealed form, so the file never grows.

### Construction configs (`config.py`)

Every ORAM/OMAP is built from a frozen, keyword-only config whose hierarchy mirrors the scheme
hierarchy and holds each scheme's defaults:

- `OramConfig` (`num_data, data_size, client, name, filename, bucket_size, stash_scale, encryptor`) →
  `PathOramConfig`, `StaticOramConfig`, `MulPathOramConfig`, `RecursiveOramConfig`, `CounterOramConfig`
  → `DaOramConfig`/`FreecursiveOramConfig`.
- `OmapConfig(OramConfig)` adds `key_size` and `distinguishable` → AVL/B+ configs (and their cached
  variants) and `GroupOmapConfig`. `OramOstOmapConfig` stands alone (only `num_data`).

Configs validate in `__post_init__`. Dataclasses don't chain `__post_init__`, so every subclass that
adds a constraint calls `super().__post_init__()` first.

The config is the single source of truth. `TreeStorageBase` exposes each field as a read-only private
property (`self._num_data`, `self._client`, …). Derived values (`_level`, `_leaf_range`, `_disk_size`)
are computed once in `__init__` because they are hot. Objects that aren't configuration (`StaticOram`'s
`prf`, `GroupOmap`'s `upper_oram`, `OramOstOmap`'s `ost`/`oram`) are separate constructor arguments.
Derive variants with `dataclasses.replace`.

**Recursion plumbing stays out of the config.** `_is_pos_map`, `_last_oram_data`, and
`_last_oram_level` are keyword-only private constructor arguments. A recursive scheme builds each
position-map child with `replace(self._config, num_data=…, name=f"{name}_pos_map_{i}", …)`. The child
shares the parent's client and owns that name, which is also its storage label. Each child drives its
own per-level I/O (`_access_pos_map_level`); the parent only samples leaves and threads them down the chain.

---

## 4. ORAM layer (`src/oblivlib/oram/`)

### `TreeStorageBase`

`src/oblivlib/dependency/tree_storage_base.py`, generic over its config type. It is the shared core of
`TreeBaseOram` and the ODS base `OstBaseOmap`: config accessors, level/leaf-range/stash-size math,
padded sizing, `_get_new_leaf`, the stash with its capacity check (`_check_stash`), block-major eviction
(`_evict_stash`), and per-bucket path encrypt/decrypt driven by a `BlockCodec`
(§8). It lives in `dependency/` so both higher layers depend downward on it rather than on each other.

### `TreeBaseOram` and the access protocol

Adds the position map and eviction. `operate_on_key(key, value=UNSET)` returns the value *before* any
write:

1. Look up the key's leaf and remap it to a fresh random leaf.
2. Read the old leaf's path (round trip 1).
3. Pull the path's real blocks into the stash; find the key, read it, optionally overwrite, and remap it.
4. Evict the stash onto the same path and write it back (round trip 2).

`operate_on_key_without_eviction` + `eviction_with_update_stash` split this so callers can defer or
batch the write-back.

**Eviction is block-major and that is deliberate.** Each stash block goes to its deepest legal bucket,
bubbling up if full. This matches the textbook bucket-major sweep's overflow exactly (both are maximal
placements on a laminar matroid) and is ~15% faster at realistic stash sizes. Do not switch.

### Schemes

| Scheme | Idea |
|---|---|
| `PathOram` | Random leaf reassignment on every access. |
| `MulPathOram` | Reads/evicts several paths per batch (`operate_on_keys*`, `eviction_for_mul_keys`). `stash_scale_multiplier` is baked into `stash_scale` at construction. |
| `StaticOram` | Leaves fixed by `PRF(key)`: the "new" leaf equals the current one. |
| `RecursivePathOram` | Position map compressed into a chain of smaller ORAMs, each block holding `compression_ratio` child leaves; only the smallest map (≤ `on_chip_mem`) stays on chip. A short final block is padded with random leaves. |
| `DAOram`, `FreecursiveOram` | Leaves are `PRF(key ‖ GC ‖ IC)`; position-map blocks hold counters instead of leaves (below). |

### Counter blocks (DA / Freecursive)

A position-map block covers `num_ic` children and stores a group counter (GC) plus one individual
counter (IC) per child. An access bumps the child's IC and recomputes its leaf from the PRF. When ICs
would overflow, the block **resets**: GC increments, ICs zero, and every child's leaf is recomputed.

- **Freecursive** layout is `GC ‖ IC×num_ic`. `reset_method="prob"` resets with probability
  `reset_prob` (default `1/num_ic`, drawn from the OS CSPRNG) and treats an overflow as an error.
  `"hard"` resets only on overflow, which is the insecure original scheme. A reset fans out into chunks
  of two child paths read together. Slots past the end of the data read a random path so the data
  boundary stays hidden. When a reset is outstanding, `eviction_with_update_stash(execute=False)` still
  executes internally, because later reset chunks depend on the earlier write-backs.
- **DAOram** layout is `GC ‖ IC×num_ic ‖ indicator×num_ic`. Instead of resetting all children at once,
  it de-amortizes: an indicator bit marks a child whose backup count is still pending, and each access
  carries at most one pending reset alongside it. A consumed backup clears only that indicator. An
  overflow bumps GC and sets every other child's indicator. **Every access reads two paths**, the data
  path and a reset path (random when none is due), so a reset is indistinguishable from no reset.

The on-chip and in-ORAM variants of the counter update, and the prob/hard variants, are deliberately
duplicated rather than shared: they differ in obliviousness-critical details. Reset state moves through
`NamedTuple` records (`ResetLeaf`, `ProcessedData`, `ResetEntry`, `UnpackedReset`). `ResetLeaf`'s field
is `offset` because a NamedTuple field cannot shadow `tuple.index`.

---

## 5. OMAP layer (`src/oblivlib/omap/`)

### Families

- **ODS maps.** `OstBaseOmap` (a `TreeStorageBase`) with `AVLOmap`, `BPlusOmap`, and the cached
  `AVLOmapCached`/`BPlusOmapCached`. Each holds a `root` pointer `(key, leaf)`, a per-op `local` set of
  downloaded nodes (`LocalNodesBase` plus scheme `LocalNodes`), and the stash, and exposes `search`,
  `insert`, and `delete`. `search(key, value)` returns the old value and writes `value` when it is not
  `None`; unlike the ORAMs, an ODS cannot write `None`. Insert assumes the key is absent: duplicate keys
  are undefined behaviour.
- **Composed maps.** `OramOstOmap` (VLDB 2025) hashes each key with a PRF into an ORAM slot that stores
  the root of that slot's ODS tree. An op fetches the root, runs the ODS op, and writes the root back.
  It works with any `TreeBaseOram` × `OstBaseOmap`, and calls `update_mul_tree_height` so the ODS sizes
  its budgets for one small tree per slot. `GroupOmap` hashes keys into buckets: an upper ORAM stores
  per-bucket metadata `pickle((count, seed, keys))` and a lower `MulPathOram` stores pairs at
  `PRF(seed ‖ key)`. A search reads the whole bucket and reshuffles it under a new seed. An insert
  updates the key's single pre-existing lower block in place, because appending a second block would let
  a later read return the stale placeholder. The upper ORAM must be at least
  `GroupOmap.upper_oram_data_size(num_data, key_size)` wide; construction checks this.

### Obliviousness model

Every op pads to a fixed round budget: real fetches are counted in `_op_rounds`, then `_pad_to(budget)`
adds dummy read+evict rounds. `_short_circuit_read` routes dummy (`key is None`) and empty-tree ops
through the same padding, so a hit, a miss, a dummy, and an empty map all look identical. Budgets come
from `_op_round_bounds` as functions of the height bound `h`. With the default
`distinguishable=False`, every op pads to the largest bound so the op type is hidden; `True` lets each
op use its own bound.

Bounds count fetch rounds plus matching eviction rounds:

- **AVL:** search/insert `2h+1`. Insert needs at most one rotation, and its nodes are already on the
  descent path. Delete is `6h`: locate the node plus its successor (one downward path, ≤ h), a rebalance
  that can rotate at every ancestor fetching the off-path child and grandchild (≤ 2h), and evictions.
  Two-children deletes replace the node from the taller subtree, the same choice as `AVLTree.delete`.
- **B+:** search/insert `2h+1`. Delete is `4h`: it prefetches the path child *and* one sibling (left
  preferred) at every level whether or not an underflow happens, so the read count never reveals a
  borrow or merge.

`search` streams: it re-homes and evicts each node before reading the next, so `local` stays O(1).

**Height bounds.** AVL uses `max(1, ceil(1.44·log₂ n))` and B+ uses
`max(1, ceil(log_⌈order/2⌉ n))`, both floored at 1 so single-element maps keep a positive budget. In
multi-tree mode `update_mul_tree_height` first bounds the per-slot tree size with the Lambert-W
max-load formula of eprint 2021/1280 at a 2⁻¹²⁸ overflow probability, then applies the same height
bound. The AVL bound is safe but runs 1–2 levels above the exact minimal-node recurrence.

**Cached variants are deliberately not per-op oblivious.** They serve nodes left in the stash by
earlier ops without a server round, keep the visited path in `local` until the next op, and pad to
`h` rounds (`2h` for AVL delete). They are meant for use inside a larger oblivious composition.
`BPlusOmapCached.delete` reads each level's child and sibling in one batched round (`h` rounds, same
bandwidth as `2h−1` paths).

### Plaintext reference trees (`dependency/avl_tree.py`, `bplus_tree.py`)

`AVLTree`/`BPlusTree` build the initial ODS storage (`get_data_list`) and are the standard the oblivious
ports mirror. Each keeps a recursive twin of `insert`/`delete` that tests cross-check structurally.

- AVL routes equal-or-larger keys right.
- B+ leaf splits keep the median in the right half and copy it up as the separator. Internal splits
  move the median up and carry the extra child pointer. The median is read before the split mutates the
  node. Underflow repair uses a single sibling (left preferred): borrow if it can spare a key, else merge
  with the left node absorbing the right.
- `multi_search`/`multi_insert` are level-synchronized: every cursor advances one level per round, so a
  batch costs at most `h` rounds of growing width. `multi_insert` fetches the union of insertion paths
  first, then replays single inserts against that partial tree, raising if a node outside it is touched.

---

## 6. Graph layer (`src/oblivlib/graph/`)

`GraphOS` (VLDB 2024) and `Grove` build on `MulPathOram` + `AVLOmapCached`, using
`dependency/graph.py` for graph generation. Broken against the current API (§10).

## 7. SORAM (`src/oblivlib/soram/`)

A weaker-threat-model ORAM that hides access patterns only over windows of `c` consecutive operations.
Excluded from the default test run and from type checking (§10).

---

## 8. Crypto (`src/oblivlib/dependency/crypto.py`)

- `AesGcm`: `nonce(12) ‖ ciphertext ‖ tag(16)`. GCM adds no padding, so length is plaintext + 28.
- `Blake2Prf`: keyed BLAKE2b, with `digest_mod_n` for leaf derivation.
- `FeistelPrp`: a 4-round balanced Feistel (the Luby–Rackoff bound for a strong PRP) over an even bit
  width. The SHA-256 round function clones a key-seeded hash per round, which is equivalent to hashing
  `key ‖ round ‖ value`. Non-power-of-2 domains use cycle-walking, which stays a bijection.

**Encryption is per bucket.** The init seal (`Storage.encrypt`) and per-op `_encrypt_path_data` both
concatenate `bucket_size` fixed-width blocks (dummies included) and encrypt once via
`Helper.encrypt_bucket`, so there is one nonce+tag per bucket. GCM authentication is defense in depth
under the honest-but-curious model, not malicious-server integrity: it has no replay or freshness
protection. Random 96-bit nonces cap safe use at ~2³² encryptions per key.

**Per-block serialization is a `BlockCodec`** (`codec.py`). `DefaultCodec` stores the value verbatim.
`NodeCodec(block_size, AVLData | BPlusData)` stores a node value as its own pickle and rebuilds it on
load. Codecs never mutate the live block. A new value-carrying scheme overrides `_codec` instead of
reimplementing path encryption.

`encryptor=None` stores plaintext and skips both steps; the tests use this for speed.

---

## 9. Invariants and contracts

- A block mapped to leaf `x` is on the root→`x` path or in the stash.
- Recursive schemes require `num_data` > on-chip size (`on_chip_mem` for DA/Recursive, `on_chip_size`
  for Freecursive); the constructor raises otherwise.
- In encrypted or file mode a value's pickle must fit `data_size`.
- The init seal and `_encrypt_path_data` must produce identical per-bucket layouts: same
  `codec.block_size` (`_dumped_data_size` for `DefaultCodec`, `_max_block_size` for `NodeCodec`) and
  same per-block bytes. That size is also the `data_size` each scheme passes to `Storage`.
- Every scheme on a shared client has a distinct `name`.
- ORAM: `operate_on_key(key)` reads, `operate_on_key(key, None)` writes `None`, and the return value is
  always the pre-write value.
- A real op never exceeds its round budget: `_perform_dummy_operation` raises on a negative pad count,
  so a broken bound fails loudly rather than leaking.

---

## 10. Known issues

- **`FlexibleBinaryTree`** is orphaned (kept for soram). Its `scale_up`/`scale_down` behaviour is
  characterized by tests, not specified; validate it against real requirements when soram adopts it.
  Its leaf labels are `(leaf, level)` tuples stored in `Data.leaf` (typed `int`) through a cast. Its
  `get_cross_index` takes raw storage indices at equal depth.
- **`graph/` and `soram/`** call ORAM/OMAP APIs that no longer exist (the old batch-path primitives,
  `read_mul_query`, `search_with_meta`). They import but fail on first use, have no tests, and are
  excluded from `pyrightconfig.json` pending a port.
- **Type checking:** basedpyright `recommended` is clean across `dependency`/`oram`/`omap` with no
  `type: ignore`. `OstBaseOmap` is generic over its `LocalNodes` container rather than narrowing a base
  attribute.
- **Tree OMAPs in plaintext file mode are broken.** `Storage` serializes file slots with
  `Data.dump_pad`, not the scheme's codec, so an ODS node object can overflow a slot sized for the
  byte-encoded node. AVL fails; B+ passes only through slack. The encrypted path works because
  `get_data_list(encryption=True)` pre-encodes node values. The tests mark this as `xfail`.
- **Wire format is pickle:** fine under honest-but-curious, but an RCE vector against a malicious peer.
  Bandwidth figures include pickle framing.
- **Two round trips per ORAM access.** The deferred-eviction primitives could piggyback access N's
  write-back on access N+1's read; the default path does not.

---

## 11. Tests (`tests/`)

- `tests/conftest.py`: the `num_data` fixture (default `2**12`, overridable via the `NUM_DATA` env var,
  which avoids needing a rootdir `pytest_addoption`), `client`, `encryptor`, `test_file`, and the soram
  exclusion.
- `tests/dependency/`: `make_search_tree` parametrizes one behavioural suite
  (`test_search_tree_common.py`) over AVL and B+. Per-tree files hold the structural invariant validators
  and recursive/iterative and batched/sequential cross-checks. Storage tests are parametrized over the
  four backends. Test order mirrors source order. Trivial helpers covered transitively get no dedicated
  test.
- `tests/oram/`: `make_oram` parametrizes over `ORAM_SPECS` (Freecursive twice, prob/hard) and
  `storage_kwargs` over the four backends. Adding one `ORAM_SPECS` line enrols a scheme in the whole
  common suite. `MulPathOram`'s batch API, edge cases, and the remote protocol (through an in-process
  pickling loopback) have their own files.
- `tests/omap/`: `omap_spec` parametrizes over `OMAP_SPECS`, each entry declaring its guarantees
  (`per_op_oblivious`, `delete_oblivious`, …). The obliviousness tests draw from filtered fixtures, so
  they are *collected* only for variants that claim the property. Cached variants are excluded by
  construction, so there are no standing skips. Composed maps have their own files.

Uncovered: an empty `operate_on_keys({})` batch and a forced stash-overflow `MemoryError`.
