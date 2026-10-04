# oblivlib — Design Knowledge Base

How the library is put together, for anyone (human or agent) about to change it. The source carries no
comments by convention: any rationale the code cannot express lives here. Paper-to-file mapping is in
`README.md`; commands and conventions are in `CLAUDE.md`.

---

## 1. Scope and threat model

`oblivlib` implements oblivious data structures (ORAM and OMAP) for a
client/server deployment:

- The **client** is trusted, holds all secret state (position map, stash, keys), and may compute
  non-obliviously.
- The **server** is **honest-but-curious**: it follows the protocol, stores only encrypted blocks, and
  answers batched read/write requests. It must not learn the access pattern.

Because the server is not malicious, the wire format does not defend against hostile responses (§8).

---

## 2. The storage seam (`InteractServer`)

`src/oblivlib/dependency/interact_server.py`. Every storage access goes through this interface. Schemes
receive it as `client=`: it is the client-side handle *to* the server, not the server.

| Class | Role |
|---|---|
| `InteractLocalServer` | Storage in-process. Used by tests and local runs. |
| `InteractRemoteServer` | Client side: ships batched queries over a socket. |
| `RemoteServer` | Server side: runs received queries on local storage. Subclasses `InteractLocalServer`. |

**Query batching.** No call does I/O immediately. Queries accumulate in four label-keyed buffers:
`add_read_path`, `add_read_list`, `add_write_path`, and `add_write_list`. `execute()` takes them as one
`Request` (`_take_request` snapshots the buffers and rebinds them empty, so a failed `execute()` never
leaks queries into the next one) and returns `ExecuteResult(results, error)`: results keyed by label,
and `error` holding the exception a failed `execute()` caught (else `None`).

- Within one `execute()` the server applies path writes, then list ops, then reads. A later path write to
  the same bucket overwrites an earlier one.
- Reads are deduplicated. `add_read_list(label, None)` reads the whole list.
- List writes are an ordered sequence of `ListOp`s (`ListWrite(index, value)`, `ListPushFront(value)`,
  `ListPopBack()`, in `types.py`), applied in staging order, so several pushes in one batch all land.
  `ListWrite` rejects a negative index at construction.
- The local server pickles the `Request` on the way in and the `ExecuteResult` on the way out, exactly as
  the wire does. Results and stored writes are therefore copies: a client edit to a read bucket or a
  written block never reaches server storage, and a scheme that only works through that aliasing is
  broken under `InteractRemoteServer`.
- Bandwidth (`get_bandwidth`/`reset_bandwidth`) is `len(pickle.dumps(...))` of the `Request` and the
  `ExecuteResult`, counted by the client-side handle, so local and remote report identical numbers.
  `RemoteServer` does not count. For experiments only.
- Read results through `ExecuteResult.require(label)`: on a failed `execute()` it re-raises the original
  error object (e.g. `UnknownLabelError` for a label the server does not host), and a label with no result
  raises `MissingResultError`. Library errors subclass `OblivlibError` (`errors.py`) and keep the default
  single-message constructor so they survive the pickle trip back from a remote server.

`init_storage` takes `dict[label, BinaryTree | list]` and routes each store into the server's tree or
list table; it takes ownership without copying. It validates the whole batch before hosting anything:
a label already hosted raises `DuplicateLabelError`, and a store of any other type raises `TypeError`.
Schemes sharing one client must use distinct names and distinct `filename`s: a file-backed `Storage`
truncates its file on construction.

**Transport.** `sockets.py`: `ZMQSocket` (ZeroMQ `REQ`/`REP`, pickle framing). Two messages, both
answered with an `ExecuteResult`: `("init", storage)` and `("execute", Request)`.
`RemoteServer._process_request` turns any failure (init, execute, or an unknown command) into an
`ExecuteResult(error=...)` reply, so `run()` never dies and the `REQ` client never blocks;
`InteractRemoteServer.init_storage` re-raises a reply's error.

---

## 3. Data and storage layer

### Blocks

`Data(key, leaf, value)` (`types.py`). A dummy block has `key is None`. `UNSET` (`types.py`)
distinguishes "read" from "write `None`": `None` is a writable value.

`Data`, `AVLData`, and `BPlusData` share one serializer, the `FieldTuplePickle` mixin: `dump()` pickles
the tuple of the dataclass's fields in declaration order and `load()` rebuilds the instance from it.
Reordering, adding, or removing a field changes the stored byte format.

### Padding and sizing (encryption or file storage only)

- `dump_pad(length)` = `pickle.dumps((key, leaf, value))` + zero padding. Fields are pickled shallowly
  (not via `dataclasses.astuple`, which would flatten a dataclass value into a tuple). There is no length header:
  pickle is self-delimiting and always ends in STOP (`0x2e`, never `0x00`), so a padded slot loads with
  the plain `Data.load`.
- `_dumped_data_size` (a cached property, always available) is the exact worst-case pickle for
  keys/leaves/values at the configured sizes. A value whose pickle exceeds it makes `dump_pad` raise.
- The scheme's codec fixes the slot width (`codec.block_size`); `Storage` derives its own row size from
  it (below).

Memory + plaintext mode stores `Data` objects as-is, so values may be any picklable object; the codec is
unused there.

### Heap-index math (`heap_index.py`)

Trees are complete binary trees flattened into heap order: the root is index 0 and node `i`'s parent is
`(i - 1) // 2`. All tree math is integer-only module functions in `heap_index.py` (imported from the
submodule, not re-exported): `compute_level(num_data) = (num_data - 1).bit_length() + 1` (the smallest
level whose `2**(level-1)` leaves cover `num_data`), `leaf_index(leaf, level) = leaf + 2**(level-1) - 1`,
`parent`, `path_to_root` (node first), `union_of_paths` (deduplicated, deepest first),
`empty_path`, and `fill_data_to_path`.

- `leaf_lca(leaf_a, leaf_b, level)` is O(1). Two leaves at the same depth have 1-based heap indices
  `leaf + 2**(level-1)` of equal bit length, and an ancestor's 1-based index is a binary prefix of its
  descendant's. The highest set bit of `a ^ b` marks the first level where the two root-to-leaf paths
  diverge, so `a >> (a ^ b).bit_length()` is their longest common prefix, the lowest common ancestor
  (minus 1 for the 0-based index). This replaces an O(depth) parent walk in the inner loop of stash
  eviction. It only holds for nodes at equal depth, which is all eviction needs.
- `fill_data_to_path` places a block at the deepest bucket on both its own path and the target path
  set, bubbling up toward the root if full.

### `BinaryTree` and `Storage`

`BinaryTree` is a fixed-level heap of `2**level - 1` buckets over `Storage`, read and written one set of
leaf paths at a time.

- Storage hands back bucket copies in file mode, so an in-place edit (`fill_data_to_storage_leaf`)
  reads the bucket, mutates it, and writes it back explicitly.
- `BinaryTree` allocates all `2**level - 1` buckets eagerly. Never construct one at large `num_data`
  just to read its level; use `heap_index.compute_level`.

`Storage(size, bucket_size, codec, encryptor=None, filename=None)` owns the scheme's `BlockCodec` and
fronts two backends: `_MemoryBackend` (a list of buckets) and `_FileBackend` (a pre-allocated
fixed-row file, truncated on open).

- The encryptor is **never stored**. File mode uses it only to size a sealed row
  (`ciphertext_length(bucket_size * codec.block_size)`); `seal(encryptor)` takes it again. A pickled
  `Storage` (shipped to a remote server by `init_storage`) therefore never carries the key, and the
  unpicklable `AesGcm` never needs to be.
- The file backend is two-phase. While the tree is filled, rows hold `bucket_size` codec-encoded
  plaintext slots. `seal()` then streams one row at a time, sealing each into a single blob through
  `codec.seal_bucket` and flipping `_sealed`, so trees larger than RAM never materialize. Rows are sized
  for the sealed form, so the file never grows.
- Writes are bounded: a plaintext file row rejects more than `bucket_size` blocks or a block wider than
  its slot (`dump_pad` raises), and a sealed row rejects a blob longer than the row, so an oversized
  write raises instead of spilling into the neighbouring row.
- `resize(size)` extends or truncates in place (memory: the list; file: `truncate`, so the heap prefix
  stays put). New rows read as `[]`.

`TreeStorageBase._build_tree(blocks)` is the one tree-construction path: it builds the `BinaryTree` with
the scheme's codec, fills each block onto its leaf path (a block whose path is already full goes to the
stash, as in Path ORAM), and seals when encrypted. `TreeBaseOram._initial_blocks` supplies the ORAM
blocks (`StaticOram` overrides only that), the recursive schemes build each position-map tree with the
child's own `_build_tree`, and the ODS init paths collect their blocks and call it once.

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
property (`self._num_data`, `self._client`, …). Derived values (`_level`, `_leaf_range`) are computed
once in `__init__` because they are hot. Objects that aren't configuration (`StaticOram`'s
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
padded sizing, `_get_new_leaf`, the stash with its capacity check (`_check_stash`, raising
`StashOverflowError`, which is also a `MemoryError`), tree construction (`_build_tree`), block-major
eviction (`_evict_stash`), and per-bucket path encrypt/decrypt driven by a `BlockCodec` (§6). It lives in `dependency/` so both higher layers depend downward on it rather than on each other.

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
- **Key encoding.** Both composed maps hash keys through `crypto.key_to_bytes`: an int is 16-byte
  big-endian two's complement (so negative keys work, and an int outside [-2¹²⁷, 2¹²⁷) raises
  `OverflowError`), a str is UTF-8, and bytes pass through. `GroupOmap` falls back to `str(key)` for any
  other key type.

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
max-load formula of eprint 2021/1280 at a 2⁻¹²⁸ overflow probability (`load_bound.max_bucket_load`,
which `GroupOmap` also uses for its bucket bound; `lambert_w` solves `w·eʷ = x` by Halley's method),
then applies the same height bound. The AVL bound is safe but runs 1–2 levels above the exact minimal-node recurrence.

**Cached variants are deliberately not per-op oblivious.** They serve nodes left in the stash by
earlier ops without a server round, keep the visited path in `local` until the next op, and pad to
`h` rounds (`2h` for AVL delete). They are meant for use inside a larger oblivious composition.
`BPlusOmapCached.delete` reads each level's child and sibling in one batched round (`h` rounds, same
bandwidth as `2h−1` paths).

### Plaintext reference trees (`dependency/avl_tree.py`, `bplus_tree.py`)

`AVLTree`/`BPlusTree` build the initial ODS storage (`get_data_list`) and are the standard the oblivious
ports mirror. Each keeps a recursive twin of `insert`/`delete` that tests cross-check structurally. The
non-recursive `delete`s are the templates for the oblivious deletes: they record the visited path in a
`local` list in clear phases (locate, remove, repair bottom-up). Their batched and recursive methods are
kept as templates and cross-check oracles even though only tests call them.

- An ORAM block's value is one node: `AVLData` holds the value plus each child's `(key, leaf, height)`;
  `BPlusData` holds parallel `keys`/`values` lists, where an internal node's values are child
  `(id, leaf)` pairs and a leaf's are the stored values.
- AVL routes equal-or-larger keys right. A two-children delete replaces the node with the in-order
  predecessor when the left subtree is taller, else the successor; `delete` and `recursive_delete` make
  the same choice, so they produce structurally identical trees.
- B+ leaf splits keep the median in the right half and copy it up as the separator. Internal splits
  move the median up and carry the extra child pointer. The median is read before the split mutates the
  node. Underflow repair (`_fix_underflow`) uses a single sibling (left preferred): borrow if it can
  spare a key, else merge with the left node absorbing the right. Considering exactly one sibling rather
  than the better of two is what the oblivious map follows: it prefetches just that sibling, so the work
  per level is fixed and independent of the borrow/merge outcome. Both deletes route through
  `_fix_underflow`, which is what makes them cross-checkable.
- `multi_search`/`multi_insert` are level-synchronized: every cursor advances one level per round, so a
  batch costs at most `h` rounds of growing width. `multi_search` maps absent keys to `None` (B+'s
  single-key `search` raises instead). `multi_insert` is two-phase. Phase 1 (`_collect_insert_paths`) is
  its only storage access: one batched descent gathering the union of the insertion paths into `local`.
  Phase 2 (`_insert_into_local`) replays single-key inserts against `local` only, raising if it needs a
  node outside it; nodes a split creates join `local` so later inserts can descend onto them. AVL
  rebalances per key (it cannot be batch-rebalanced in one pass), so for both trees the result equals
  sequential single inserts, the shape the oblivious port mirrors.

---

## 6. Crypto (`src/oblivlib/dependency/crypto.py`)

- `AesGcm`: `nonce(12) ‖ ciphertext ‖ tag(16)`. GCM adds no padding, so length is plaintext + 28.
- `Blake2Prf`: keyed BLAKE2b, with `digest_mod_n` for leaf derivation.
- `FeistelPrp`: a 4-round balanced Feistel (the Luby–Rackoff bound for a strong PRP) over an even bit
  width. The SHA-256 round function clones a key-seeded hash per round, which is equivalent to hashing
  `key ‖ round ‖ value`. Non-power-of-2 domains use cycle-walking, which stays a bijection.

**Encryption is per bucket.** The init seal (`Storage.seal`) and per-op `_encrypt_path_data` both go
through `BlockCodec.seal_bucket`, which concatenates up to `bucket_size` codec-encoded blocks, pads with
dummy blocks, and encrypts once, so there is one nonce+tag per bucket; `open_bucket` reverses it and
keeps only real blocks. GCM authentication is defense in depth
under the honest-but-curious model, not malicious-server integrity: it has no replay or freshness
protection. Random 96-bit nonces cap safe use at ~2³² encryptions per key.

**Per-block serialization is a `BlockCodec`** (`codec.py`). `DefaultCodec` stores the value verbatim.
`NodeCodec(block_size, AVLData | BPlusData)` stores a node value as its own pickle and rebuilds it on
load, so `get_data_list` always emits live node values and the codec encodes them at seal or write
time. Codecs never mutate the live block. A new value-carrying scheme overrides `_codec` instead of
reimplementing path encryption.

`encryptor=None` stores plaintext and skips both steps; the tests use this for speed.

---

## 7. Invariants and contracts

- A block mapped to leaf `x` is on the root→`x` path or in the stash.
- Recursive schemes require `num_data` > on-chip size (`on_chip_mem` for DA/Recursive, `on_chip_size`
  for Freecursive); the constructor raises otherwise.
- In encrypted or file mode a value's pickle must fit `data_size`.
- The init seal and `_encrypt_path_data` must produce identical per-bucket layouts. Both use the one
  `BlockCodec.seal_bucket` with the scheme's `_codec` (`_dumped_data_size` for `DefaultCodec`,
  `_max_block_size` for `NodeCodec`), and `_build_tree` hands that same codec to `Storage`.
- Every scheme on a shared client has a distinct `name`, and every file-backed scheme a distinct `filename`.
- ORAM: `operate_on_key(key)` reads, `operate_on_key(key, None)` writes `None`, and the return value is
  always the pre-write value.
- `FlexibleBinaryTree` absent nodes are zero rows and are skipped by `read_path`, so the server sees
  which nodes are empty versus present. That is the same information the write lengths already reveal,
  and the adopting scheme controls both.
- A real op never exceeds its round budget: `_perform_dummy_operation` raises on a negative pad count,
  so a broken bound fails loudly rather than leaking.

---

## 8. Known issues

- **`FlexibleBinaryTree`** (`flexible_binary_tree.py`) has no callers yet; it is the resizable tree a
  future scheme (e.g. SORAM) is meant to adopt. Its model is client-driven: leaves are int labels at
  the tree's current `level` (the caller keeps them in range), and every node is either present (a
  bucket, possibly empty) or absent. The tree tracks presence itself in a `bytearray`, one byte per
  node, that is not persisted; all nodes start present. `read_path(leaves)` returns the present nodes on
  those paths, root first. `write_path(leaves, data)` rejects an index off those paths, writes each
  bucket in `data` (marking it present), and clears every other node on the paths to `[]` (absent).
  `scale_up()` adds an absent bottom layer and `scale_down()` drops the bottom layer, raising
  `ScaleDownError` at level 1 or while any bottom node is present. Neither moves data: evacuating the
  bottom layer before shrinking is the scheme's job, done through ordinary `write_path` calls. Both
  resize `Storage` in place, so the heap prefix and its sealed blobs survive.
- **Type checking:** basedpyright `recommended` is clean across `dependency`/`oram`/`omap` with no
  `type: ignore`. `OstBaseOmap` is generic over its `LocalNodes` container rather than narrowing a base
  attribute.
- **Wire format is pickle:** fine under honest-but-curious, but an RCE vector against a malicious peer.
  Bandwidth figures include pickle framing.
- **Two round trips per ORAM access.** The deferred-eviction primitives could piggyback access N's
  write-back on access N+1's read; the default path does not.

---

## 9. Tests (`tests/`)

- `tests/conftest.py`: the `num_data` fixture (default `2**12`, overridable via the `NUM_DATA` env var,
  which avoids needing a rootdir `pytest_addoption`), `client`, `remote_client` (an
  `InteractRemoteServer` wired to a real `RemoteServer` through an in-process pickling loopback),
  `encryptor`, and `test_file`.
- `tests/dependency/`: `make_search_tree` parametrizes one behavioural suite
  (`test_search_tree_common.py`) over AVL and B+. Per-tree files hold the structural invariant validators
  and recursive/iterative and batched/sequential cross-checks. Storage tests are parametrized over the
  four backends. Test order mirrors source order. Trivial helpers covered transitively get no dedicated
  test.
- `tests/oram/`: `make_oram` parametrizes over `ORAM_SPECS` (Freecursive twice, prob/hard) and
  `storage_kwargs` over the four backends. Adding one `ORAM_SPECS` line enrols a scheme in the whole
  common suite. `MulPathOram`'s batch API, edge cases, and the remote protocol (through
  `remote_client`) have their own files.
- `tests/omap/`: `omap_spec` parametrizes over `OMAP_SPECS`, each entry declaring its guarantees
  (`per_op_oblivious`, `delete_oblivious`, …). The obliviousness tests draw from filtered fixtures, so
  they are *collected* only for variants that claim the property. Cached variants are excluded by
  construction, so there are no standing skips. Composed maps have their own files.

Uncovered: an empty `operate_on_keys({})` batch.
