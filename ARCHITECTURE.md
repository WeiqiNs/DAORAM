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

Because the server is not malicious, the protocol does not defend against hostile responses beyond
decoding every message as plain data (§2): a peer can lie, but it cannot run code.

---

## 2. Client, protocol, and server (the wire spec)

Every storage access goes through a `Client` (`src/oblivlib/dependency/client.py`). Schemes receive it
as `client=` in their config: it is the client-side handle *to* the server, not the server. The layers
under it, each owning one concern:

| Layer | Owner | Role |
|---|---|---|
| Scheme-facing API | `Client` | Stages queries, defers writes, hosts trees, keeps metrics. |
| Messages | `Backend` (`LocalBackend`, `TransportBackend`) | Delivers one request and returns its reply. |
| Bytes | `Transport` (`transport.py`) | Moves an encoded request; `ZmqTransport`, `LoopbackTransport`, `SimulatedNetwork`. |
| Engine | `StorageServer` (`storage_server.py`) | Answers requests; no transport of its own. |
| Stores | `server_stores.py` | Fixed-row trees (memory or file), variable-row trees (memory), lists. |

`Client.local(storage_dir=None)` wires a `LocalBackend` straight to a `StorageServer`: rows are
immutable `bytes`, so nothing is copied or encoded in process. `Client.connect(endpoint)` wires a
`TransportBackend` over a `ZmqTransport`; the server side runs `serve(ZmqListener(endpoint), server)`
until it has answered a `Shutdown`. A client is not thread-safe: use one per thread.

### Framing and decoding

Every message is a frozen, slotted dataclass in `protocol.py`, framed as the msgpack array
`[ClassName, *fields in declaration order]`; nested ops are framed the same way. Fields hold only ints,
str, bytes, None, lists, and nested ops. Decoding (`decode_request`, `decode_reply`) is the trust
boundary: it checks the tag against the direction's union, the arity, and every field's type against
the dataclass annotations (an int never accepts a bool), and raises `ProtocolError` for anything else,
including a pickle payload. Nothing is ever unpickled. The message set is the `Request`, `Reply`,
`WriteOp`, and `ReadOp` unions in `protocol.py`; read those for the current members. `PROTOCOL_VERSION`
is checked by `Hello`.

### Sessions and store lifecycle

Every request except `Hello` and `Shutdown` carries its session first. The client sends `Hello` lazily,
on its first request; the reply carries the session id (`secrets.token_hex(16)`) and the server's
`max_message_bytes`. Labels are namespaced per session. `CreateTree(level, row_bytes)` makes a tree of
`2**level - 1` rows: fixed-size rows when `row_bytes` is an int, variable-size (plaintext) rows when it
is `None`, which a disk-backed server refuses. `Resize` grows or shrinks a tree in place, refusing to
drop a present row (`ScaleDownError`). `WriteRange(start, rows)` writes consecutive rows (bulk
hosting). `Close` releases the session and deletes its files; `StorageServer.close()` closes every
session. A disk server keeps a tree at `storage_dir/<session>/<label as hex>.tree` and reads and writes
it with `os.pread`/`os.pwrite`.

### Batches

`Batch(session, writes, reads)` puts "writes first, then reads" in the message's structure.

- **Client side.** Path writes merge per label, a later write to a node winning; list ops append in
  staging order; path reads union per label (leaves sorted); a whole-list read absorbs index reads.
  `add_write_path` checks that the rows cover exactly the paths to their leaves, and staging an unknown
  label raises `UnknownLabelError` at once.
- **Canonical row order.** A path read or write carries its leaves, and its rows follow ascending heap
  index (root first) over the deduplicated union of their paths: `heap_index.path_indices`.
- **Absent rows.** `b""` on the wire is an absent node. A fixed-row store keeps it as an all-zero row
  and reads an all-zero row back as `b""`; a sealed row is never all zeros.
- **All or nothing.** The server validates the whole batch (`_validate_batch`) before applying any of
  it: every label exists with the right kind, every leaf is in range, each write carries one row per
  path node, every row fits its store, list ops are simulated against the running length, and list
  read indices are checked against the post-write length. Then it applies the writes in order and
  answers the reads, aligned with `Batch.reads`.
- **An empty `execute()` is still a round trip**, so the round count never depends on what was staged.

### Write deferral

`Client(defer_writes=True)`, the default, does not send an `execute()` that staged only writes: its
writes stay pending and ride on the next batch, which applies them before its own reads. Every read
therefore sees the state it would have seen undeferred, and the batch sequence is exactly the
undeferred one with each run of read-less batches folded into the next batch (leaves unioned per
label); `assert_deferral_preserves_access` in `tests/conftest.py` checks this for every scheme family.
For schemes whose operations alternate a read round and a write-back round (all of them today) the
round count halves, the same for every key and value. `flush()` sends pending writes on their own,
every lifecycle call flushes first, and `close()` flushes. An error from a deferred write surfaces on
whichever call sends it. `defer_writes=False` sends every `execute()`.

### Errors

A failure comes back as `ErrorReply(kind, message)`; `handle` never raises. `ERROR_KINDS` in
`protocol.py` maps each kind to its class in `errors.py`; `error_reply` maps an `OSError` to
`StorageError` and anything unexpected to `ServerError`; `expect_reply` raises the mapped class and
turns an unknown kind or the wrong reply type into `ProtocolError`. `Client.execute()` raises;
`ExecuteResult` holds only results, and `require(label)` raises `MissingResultError` for a label with
no result. A timeout or disconnect raises `TransportError`; `ZmqTransport` then recreates its `REQ`
socket so it stays usable, and never resends the lost request.

### Limits and metrics

`DEFAULT_MAX_MESSAGE_BYTES`, `WRITE_RANGE_CHUNK_BYTES`, and `DEFAULT_TIMEOUT_MS` live in
`protocol.py`. `handle_wire` answers an oversized message with `ErrorReply(ProtocolError)`, so a `REQ`
peer never hangs, and the client chunks `WriteRange` by bytes at
`min(WRITE_RANGE_CHUNK_BYTES, max_message_bytes // 2)`. `client.metrics` counts `rounds` (batches sent, empty ones
included), `payload_bytes` (row and list-value bytes written and read, `WriteRange` included), and
`wire_bytes` (encoded request and reply bytes, counted by `TransportBackend`; zero in process). Local
and loopback clients report identical rounds and payload. `SimulatedNetwork` wraps a transport and
accumulates modeled time, a round-trip time plus bytes over bandwidth per request, without sleeping.

### Hosting a built tree

`Client.host_tree(label, image)` hands over a tree built on the client (§3). A `MemoryImage` is
created and streamed with `WriteRange`. A `FileImage` already has the server's on-disk format, so it
can move without re-encryption: placed out of band by the client's `ship_file` (which returns its name
inside the server's storage directory) and then attached, adopted in place when the backend exposes the
server's `storage_dir` (the file is moved there and attached), or else streamed row by row. The local
build file is gone afterwards in every case. `AttachTree` accepts only a plain file name directly in
`storage_dir` (no separators, no `..`, not absolute), requires the exact size, and moves the file into
the session's directory; an in-memory server refuses it.

---

## 3. Data and storage layer

### The data contract

ORAM keys are int addresses in `[0, num_data)`; OMAP keys are `bytes` of at most `key_size`, ordered by
plain bytes comparison (OMAPs only search, insert, and delete exact keys); values are `bytes` of at most
`data_size`. Users serialize their own data. `contract.py` checks keys and values at every public entry
point, before any state changes, and raises `ContractError` (a `ValueError`) prefixed with the scheme's
identity. An ORAM address never written reads `b""`; `None` is not a value. Internal structured values
(position-map blocks, composed-map roots and metadata, ODS nodes) are msgpack; the library never pickles.

### Blocks and codecs

`Data(key, leaf, value)` (`types.py`) is a block. `Data`, `AVLData`, and `BPlusData` share the
`FieldTuple` mixin: `to_fields()` lists the dataclass's fields in declaration order and `from_fields`
rebuilds the instance, so reordering, adding, or removing a field changes the stored format.

A `BlockCodec` (`codec.py`) turns a block into the msgpack-ready `[key, leaf, value]` and back, never
touching the live block. `DefaultCodec` stores the value as is; `NodeCodec(max_block_bytes, value_cls)`
stores a node value as its field list and rebuilds it. msgpack returns lists where tuples went in, so
decoders accept lists (B+ internal values come back as `[id, leaf]`). `max_block_bytes` bounds one packed
block: `TreeStorageBase._max_block_bytes` for plain values, each ODS's `_max_block_size` for nodes, both
computed with `codec.packed_size` on a widest-case block.

### Path ciphers and rows

A `PathCipher` (`path_cipher.py`) turns a bucket into a server row and back, and is the one path both
initial building and per-op eviction go through (`TreeStorageBase._cipher`). A bucket packs as a msgpack
array of its real blocks only; there are no dummy blocks.

- `PlainPathCipher`: the row is the packed bucket, unpadded, so rows vary in length (memory servers
  only). An empty bucket packs to a non-empty row, so a present empty node stays distinct from an absent
  one.
- `SealedPathCipher`: the row is `enc(u32_be(len) ‖ packed ‖ zero padding)`, padded to the capacity of a
  full bucket of widest blocks (the largest msgpack array header plus `bucket_size * max_block_bytes`).
  Every sealed row is therefore exactly `row_bytes` long, whatever it holds. A bucket with too many
  blocks or bytes raises `RowSizeError` instead of producing a longer row.

### Heap-index math (`heap_index.py`)

Trees are complete binary trees flattened into heap order: the root is index 0 and node `i`'s parent is
`(i - 1) // 2`. All tree math is integer-only module functions in `heap_index.py` (imported from the
submodule, not re-exported): `compute_level(num_data) = (num_data - 1).bit_length() + 1` (the smallest
level whose `2**(level-1)` leaves cover `num_data`), `tree_size`, `leaf_index`, `parent`, `path_to_root`
(node first), `path_indices` (the canonical row order: deduplicated, root first), `empty_path`, and
`fill_data_to_path`.

- `leaf_lca(leaf_a, leaf_b, level)` is O(1). Two leaves at the same depth have 1-based heap indices
  `leaf + 2**(level-1)` of equal bit length, and an ancestor's 1-based index is a binary prefix of its
  descendant's. The highest set bit of `a ^ b` marks the first level where the two root-to-leaf paths
  diverge, so `a >> (a ^ b).bit_length()` is their longest common prefix, the lowest common ancestor
  (minus 1 for the 0-based index). This replaces an O(depth) parent walk in the inner loop of stash
  eviction. It only holds for nodes at equal depth, which is all eviction needs.
- `fill_data_to_path` places a block at the deepest bucket on both its own path and the target path
  set, bubbling up toward the root if full.

### Building the initial tree (`tree_builder.py`)

`build_tree(blocks, level, bucket_size, cipher, build_file)` places each block in the deepest non-full
bucket on its leaf's path and seals every bucket. Without a `build_file` the result is a `MemoryImage`
(all `2**level - 1` buckets in memory: never build one at large `num_data` just to read its level; use
`heap_index.compute_level`). With one it is a `FileImage` in the server's format: the file is truncated,
every row pre-sealed empty, and each placement reads, opens, appends, seals, and writes its row in
place, so the file never holds plaintext. Blocks whose whole path is full come back as `overflow`.

`TreeStorageBase._build_tree` is the one construction path: it builds with the scheme's cipher and
`build_file`, puts the overflow in the stash (as in Path ORAM), and checks the stash. Blocks stream in:
`TreeBaseOram._initial_blocks` yields the supplied `(key, value)` pairs (a mapping or a one-shot
iterator), tracking supplied keys in a `bytearray(num_data)` so a duplicate raises, then `b""` for every
other address; `_initial_leaf` gives each address's leaf (`StaticOram` overrides it). The recursive
schemes build and host each position-map tree with the child's own `_build_tree` and `_host_tree`, one at
a time, so the children reuse the parent's `build_file` path in turn. The ODS init paths collect their
blocks and build once.

### Server stores (`server_stores.py`)

A `TreeStore` holds `2**level - 1` rows. `FixedRowStore` keeps rows of exactly `row_bytes` in one
region, a `bytearray` or a file, and rejects any other non-empty length (`RowSizeError`);
`VariableRowStore` keeps rows of any length in memory. `resize` truncates or extends in place, so the heap
prefix and its rows survive, and refuses to drop a present row. `ListStore` holds a list of `bytes`.

### Construction configs (`config.py`)

Every ORAM/OMAP is built from a frozen, keyword-only config whose hierarchy mirrors the scheme
hierarchy and holds each scheme's defaults:

- `OramConfig` (`num_data, data_size, client, name, build_file, bucket_size, stash_scale, encryptor`)
  → `PathOramConfig`, `StaticOramConfig`, `MulPathOramConfig`, `RecursiveOramConfig`,
  `CounterOramConfig` → `DaOramConfig`/`FreecursiveOramConfig`. `build_file` requires an `encryptor`.
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
block sizing (`_max_block_bytes`), `_get_new_leaf`, the stash with its capacity check (`_check_stash`,
raising `StashOverflowError`, which is also a `MemoryError`), tree construction and hosting
(`_build_tree`, `_host_tree`), block-major eviction (`_evict_stash`, which returns sealed `PathRows`),
and the scheme's `PathCipher` (`_cipher`, built from its `_codec`; §3, §6). It lives in `dependency/` so
both higher layers depend downward on it rather than on each other.

### `TreeBaseOram` and the access protocol

Adds the position map and eviction. The public `operate_on_key`, `operate_on_key_without_eviction`,
and `eviction_with_update_stash` check the data contract and call the underscore methods each scheme
implements. `operate_on_key(key, value=UNSET)` returns the value *before* any write:

1. Look up the key's leaf and remap it to a fresh random leaf.
2. Read the old leaf's path (one batch).
3. Pull the path's real blocks into the stash; find the key, read it, optionally overwrite, and remap it.
4. Evict the stash onto the same path and stage the write-back. With write deferral (§2) it rides on the
   next operation's read batch, so an access costs one round instead of two.

`operate_on_key_without_eviction` + `eviction_with_update_stash` split this so callers can defer or
batch the write-back.

**Eviction is block-major and that is deliberate.** Each stash block goes to its deepest legal bucket,
bubbling up if full. This matches the textbook bucket-major sweep's overflow exactly (both are maximal
placements on a laminar matroid) and is ~15% faster at realistic stash sizes. Do not switch.

### Schemes

| Scheme | Idea |
|---|---|
| `PathOram` | Random leaf reassignment on every access. |
| `MulPathOram` | Reads/evicts several paths per batch (`operate_on_keys*`, `eviction_for_mul_keys`, each checking the contract before its underscore twin). `stash_scale_multiplier` is baked into `stash_scale` at construction. |
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
  `insert`, and `delete`, which check the contract and call each scheme's `_search`/`_insert`/`_delete`.
  `search(key, value)` returns the old value (`None` when absent) and writes `value` when it is not
  `None`; `key=None` is a dummy op. Insert assumes the key is absent: duplicate keys are undefined
  behaviour.
- **Composed maps.** `OramOstOmap` (VLDB 2025) hashes each key with a PRF into an ORAM slot that stores
  the root of that slot's ODS tree, as `msgpack([key, leaf])` (`b""` for an empty tree). An op fetches
  the root, runs the ODS op, and writes the root back. It works with any `TreeBaseOram` × `OstBaseOmap`,
  and calls `update_mul_tree_height` so the ODS sizes its budgets for one small tree per slot. The ORAM
  must be at least `OramOstOmap.oram_data_size(key_size)` wide; construction checks this.
- **`GroupOmap`** hashes keys into buckets. An upper ORAM stores per-bucket metadata
  `msgpack([seed, keys])`, and a lower `MulPathOram` stores each value in a block **named by its key**,
  on the path `PRF(seed ‖ key)`. A search reads every block of the key's bucket in one batch and
  reshuffles them under a new seed. An insert stashes a new block on its path through
  `MulPathOram._insert_block`, which reads and evicts one uniformly random path, so the server sees the
  same single-path access as before. GroupOmap drives the lower ORAM through its underscore
  (contract-free) methods, since the lower ORAM is internal and keyed by bytes rather than addresses. The
  lower ORAM sizes its slots for an int address, so its `data_size` is widened by `key_size` plus
  `_KEY_SLOT_OVERHEAD` (`group_omap.py`): a msgpack byte string costs its length plus at most a 5-byte
  header, against at least one byte for the int it replaces. Buckets hold at most `load_bound.max_bucket_load(num_data)` keys, and the upper ORAM must be
  at least `GroupOmap.upper_oram_data_size(num_data, key_size)` wide; construction checks this.

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
- `Blake2Prf`: keyed BLAKE2b, with `digest_mod_n` for leaf derivation. `hash_data_to_leaf` and
  `hash_data_to_map` hash `bytes` keys with it for the composed maps.
- `FeistelPrp`: a 4-round balanced Feistel (the Luby–Rackoff bound for a strong PRP) over an even bit
  width. The SHA-256 round function clones a key-seeded hash per round, which is equivalent to hashing
  `key ‖ round ‖ value`. Non-power-of-2 domains use cycle-walking, which stays a bijection.

**Encryption is per bucket.** Building the initial tree and evicting a path both seal through the
scheme's one `SealedPathCipher` (§3): the bucket's real blocks are packed as one msgpack array, length
prefixed, zero padded to the row capacity, and encrypted once, so there is one nonce+tag per bucket and
every row has the same length whatever it holds. Only the client holds the encryptor; the server stores
and returns sealed rows. GCM authentication is defense in depth under the honest-but-curious model, not
malicious-server integrity: it has no replay or freshness protection. Random 96-bit nonces cap safe use
at ~2³² encryptions per key.

`encryptor=None` stores each bucket as its plaintext msgpack row, of varying length, on a memory server
only; the tests use this for speed. A `build_file` always needs an encryptor, so plaintext never reaches
disk.

---

## 7. Invariants and contracts

- A block mapped to leaf `x` is on the root→`x` path or in the stash.
- Recursive schemes require `num_data` > on-chip size (`on_chip_mem` for DA/Recursive, `on_chip_size`
  for Freecursive); the constructor raises otherwise.
- Keys and values obey the data contract (§3); a violation raises `ContractError` before any client
  state changes or any round is spent.
- The accessed-path sequence and the round count depend only on the sequence of calls, never on keys or
  values; write deferral folds read-less batches into the next batch the same way for every input (§2).
- A sealed row is exactly `row_bytes` long whatever its occupancy or value lengths, and building and
  per-op eviction seal through the same `PathCipher`, so an initial row is indistinguishable from an
  evicted one.
- Every scheme on a shared client has a distinct `name` (its storage label).
- ORAM: `operate_on_key(key)` reads, `operate_on_key(key, value)` writes `value` (`bytes`), an address
  never written reads `b""`, and the return value is always the pre-write value.
- `FlexibleBinaryTree` absent nodes are empty rows and are skipped by `read_path`, so the server sees
  which nodes are empty versus present. That is the same information the write lengths already reveal,
  and the adopting scheme controls both.
- A real op never exceeds its round budget: `_perform_dummy_operation` raises on a negative pad count,
  so a broken bound fails loudly rather than leaking.

---

## 8. Known issues

- **`FlexibleBinaryTree`** (`flexible_binary_tree.py`) has no callers yet; it is the resizable tree a
  future scheme (e.g. SORAM) is meant to adopt. It is a client-side handle over a hosted tree
  (`FlexibleBinaryTree.create` builds and hosts it). Its model is client-driven: leaves are int labels at
  the tree's current `level` (the caller keeps them in range), and every node is either present (a
  bucket, possibly empty) or absent (an empty row); all nodes start present. `read_path(leaves)` returns
  the present nodes on those paths, root first. `write_path(leaves, data)` rejects an index off those
  paths, writes each bucket in `data`, and writes every other node on the paths as absent. `scale_up()`
  adds an absent bottom layer and `scale_down()` drops the bottom layer, both through the server's
  `Resize`, which raises `ScaleDownError` at level 1 or while any bottom node is present. Neither moves
  data: evacuating the bottom layer before shrinking is the scheme's job, done through ordinary
  `write_path` calls.
- **Type checking:** basedpyright `recommended` is clean across `dependency`/`oram`/`omap` with no
  `type: ignore`. `OstBaseOmap` is generic over its `LocalNodes` container rather than narrowing a base
  attribute. msgpack ships no stubs, so `typings/msgpack/__init__.pyi` declares the calls the library
  makes.
- **GroupOmap leaks bucket occupancy:** a search reads one lower path per key in the bucket, so the
  number of lower paths depends on how full the bucket is.
- **B+ block ids can outgrow their size bound:** `BPlusOmap._max_block_size` assumes ids below
  `num_data`, but deletes never recycle ids. The row capacity's slack has absorbed it so far.
- **Composed-map capacity errors** (a full GroupOmap bucket) raise a bare `MemoryError` after the upper
  ORAM read, leaving that access half done.
- **Not yet built:** a round driver and handle API for schemes, retrying or resending a lost batch, and
  persistence (reattaching a server's trees after a restart). `GroupOmap` ignores `build_file`, and a
  path or list read result is untyped (`ExecuteResult.require` returns `Any`).

---

## 9. Tests (`tests/`)

- `tests/conftest.py`: the `num_data` fixture (default `2**12`, overridable via the `NUM_DATA` env var,
  which avoids needing a rootdir `pytest_addoption`), `client` (`Client.local()`), `remote_client` (a
  `Client` over `TransportBackend(LoopbackTransport(StorageServer()))`, the full wire path in process),
  `recorded_client` (a factory for a client over a `RecordingBackend` that records every batch),
  `assert_deferral_preserves_access` (runs a seeded workload with deferral off and on and compares the
  batches), `encryptor`, and `test_file`.
- `tests/dependency/`: `make_search_tree` parametrizes one behavioural suite
  (`test_search_tree_common.py`) over AVL and B+. Per-tree files hold the structural invariant validators
  and recursive/iterative and batched/sequential cross-checks. The protocol, server, client, transport,
  cipher, and builder each have their own file; `FlexibleBinaryTree` runs over memory-plain,
  memory-encrypted, disk-encrypted, and loopback-encrypted servers. Test order mirrors source order.
  Trivial helpers covered transitively get no dedicated test.
- `tests/oram/`: `make_oram` parametrizes over `ORAM_SPECS` (Freecursive twice, prob/hard) and
  `storage_kwargs` over plaintext memory, encrypted memory, and an encrypted `build_file`. Adding one
  `ORAM_SPECS` line enrols a scheme in the whole common suite. `MulPathOram`'s batch API, edge cases,
  and the remote protocol and handover modes have their own files.
- `tests/omap/`: `omap_spec` parametrizes over `OMAP_SPECS`, each entry declaring its guarantees
  (`per_op_oblivious`, `delete_oblivious`, …). The obliviousness tests draw from filtered fixtures, so
  they are *collected* only for variants that claim the property, and count path reads through a
  recording client. Cached variants are excluded by construction, so there are no standing skips.
  Composed maps have their own files.

Uncovered: an empty `operate_on_keys({})` batch.
