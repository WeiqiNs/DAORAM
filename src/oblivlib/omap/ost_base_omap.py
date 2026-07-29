"""Parent class for the tree-structured ODS omaps.

Adds the ODS-specific layer (root/local bookkeeping, tree traversal, padding budgets) on top of the
shared ``TreeStorageBase`` (config accessors, level/stash math, random leaves, path encryption).
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, cast, override

from oblivlib.dependency import BinaryTree, Data, KVPair, PathData
from oblivlib.dependency.config import OmapConfig
from oblivlib.dependency.tree_storage_base import TreeStorageBase
from oblivlib.omap.base_omap import BaseOmap

# The (data key, path) pointer to a tree root.
ROOT = tuple[Any, int]
# Input key-value pairs.
KV_LIST = list[tuple[Any, Any]]


# LocalNodesBase's type parameter is the typed node view (AVLNode / BPlusNode Protocol) a concrete
# LocalNodes container exposes.
class LocalNodesBase[NodeT]:
    """Shared bookkeeping for the nodes downloaded into ``local`` during one tree-ODS operation: the
    node dict, parent links, traversal path, and root key. Concrete subclasses (AVL / B+ ``LocalNodes``)
    add their scheme-specific tracking and parametrize ``NodeT`` with their typed node view; nodes are
    stored as plain ``Data`` and cast to that view."""

    def __init__(self) -> None:
        self.nodes: dict[Any, Data] = {}  # key -> Data node
        self.parent_of: dict[Any, Any] = {}  # key -> parent_key
        self.root_key: Any = None
        self.path: list[Any] = []  # keys in traversal order (root to current)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return len(self.nodes) > 0

    def add(self, node: Data, parent_key: Any = None, child_index: int | None = None) -> None:
        # child_index is part of the shared signature so the base move methods pass it uniformly; only
        # the B+ override records it, so the base discards it.
        del child_index
        self.nodes[node.key] = node
        self.parent_of[node.key] = parent_key
        self.path.append(node.key)
        if parent_key is None:
            self.root_key = node.key

    def get(self, key: Any) -> NodeT | None:
        return cast(NodeT | None, self.nodes.get(key))

    def require(self, key: Any) -> NodeT:
        """Get a node known to be loaded; raise if it is genuinely absent."""
        node = self.nodes.get(key)
        if node is None:
            raise KeyError(f"Node {key} is not loaded in local.")
        return cast(NodeT, node)

    def get_parent(self, key: Any) -> NodeT | None:
        parent_key = self.parent_of.get(key)
        return cast(NodeT | None, self.nodes.get(parent_key)) if parent_key is not None else None

    def get_parent_key(self, key: Any) -> Any:
        return self.parent_of.get(key)

    def get_root(self) -> NodeT | None:
        return cast(NodeT | None, self.nodes.get(self.root_key)) if self.root_key is not None else None

    def require_root(self) -> NodeT:
        """Get the root node known to exist; raise if it is genuinely absent."""
        node = self.nodes.get(self.root_key) if self.root_key is not None else None
        if node is None:
            raise KeyError("The local has no root node.")
        return cast(NodeT, node)

    def to_list(self) -> list[Data]:
        return list(self.nodes.values())

    def remove(self, key: Any) -> Data | None:
        node = self.nodes.pop(key, None)
        if node:
            self.parent_of.pop(key, None)
            if key in self.path:
                self.path.remove(key)
            if self.root_key == key:
                self.root_key = None
        return node

    def clear(self) -> None:
        self.nodes.clear()
        self.parent_of.clear()
        self.root_key = None
        self.path.clear()


# OstBaseOmap's type parameters: OmapConfigT (the scheme's config) and LocalT (the scheme-specific node
# container ``_local`` holds, bound to LocalNodesBase so the base can type it without knowing the concrete
# scheme). Each scheme re-declares both (see AVLOmap / BPlusOmap).
class OstBaseOmap[OmapConfigT: OmapConfig, LocalT: LocalNodesBase[Any]](TreeStorageBase[OmapConfigT], BaseOmap, ABC):
    def __init__(self, config: OmapConfigT):
        super().__init__(config)

        # ODS-specific runtime state: the root pointer, the nodes downloaded into local this op, and the
        # real-round counter (excludes dummy padding; the final pad tops it up to the op budget).
        self._root: ROOT | None = None
        self._local: LocalT = self._new_local()
        self._op_rounds: int = 0

    @abstractmethod
    def _new_local(self) -> LocalT:
        """Create this scheme's empty local-node container (its concrete ``LocalNodes``)."""
        raise NotImplementedError

    # Scheme-specific construction parameters — read-only views onto the frozen config (see OmapConfig).
    @property
    def _key_size(self) -> int:
        return self._config.key_size

    @property
    def _distinguishable(self) -> bool:
        """Whether the operation type may leak (each op pads to its own bound) vs. fully oblivious."""
        return self._config.distinguishable

    @property
    def root(self) -> ROOT | None:
        return self._root

    @root.setter
    def root(self, root: ROOT | None) -> None:
        self._root = root

    @abstractmethod
    def update_mul_tree_height(self, num_tree: int) -> None:
        """Re-set the per-tree height for an ODS holding ``num_tree`` trees (one per upper-ORAM datum)."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def _max_block_size(self) -> int:
        """Bytes of the widest data block stored in the ORAM; each scheme overrides with its own."""
        raise NotImplementedError

    def _find_in_stash(self, key: Any) -> int:
        """Index of the node for ``key`` in the stash, or -1 if absent."""
        for i, node in enumerate(self._stash):
            if node.key == key:
                return i
        return -1

    def _flush_local_to_stash(self) -> None:
        self._stash += self._local.to_list()
        self._local.clear()

    def _move_node_to_local(
        self, key: Any, leaf: int | None, parent_key: Any = None, child_index: int | None = None
    ) -> None:
        """Read ``leaf``'s path, move the block for ``key`` into local, then evict and write back."""
        # A node being fetched always lives on a real path (child pointers carry a concrete leaf).
        assert leaf is not None
        self._move_node_to_local_without_eviction(key=key, leaf=leaf, parent_key=parent_key, child_index=child_index)

        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
        self._client.execute()

    def _move_node_to_local_without_eviction(
        self, key: Any, leaf: int | None, parent_key: Any = None, child_index: int | None = None
    ) -> None:
        """Like ``_move_node_to_local`` but without the eviction/write-back (used by the streaming ``search``)."""
        found = False
        to_index = len(self._stash)

        # This is one real server round; count it so the final pad knows how many remain.
        self._op_rounds += 1

        assert leaf is not None
        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)
        path = self._decrypt_path_data(path=path_data)

        # Keep the requested block in local; route every other block on the path to the stash.
        for bucket in path.values():
            for data in bucket:
                if data.key == key:
                    self._local.add(node=data, parent_key=parent_key, child_index=child_index)
                    found = True
                else:
                    self._stash.append(data)

        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # Not found on the fetched path -- it must already be in the stash from an earlier op.
        if not found:
            stash_idx = self._find_in_stash(key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash[stash_idx], parent_key=parent_key, child_index=child_index)
                del self._stash[stash_idx]
                return

            raise KeyError(f"The search key {key} is not found.")

    def _evict_stash(self, leaves: list[int]) -> PathData:
        """Evict stash blocks onto the given paths; blocks that don't fit stay in the stash."""
        temp_stash = []

        path = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        for data in self._stash:
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            if not inserted:
                temp_stash.append(data)

        self._stash = temp_stash

        return self._encrypt_path_data(path=path)

    def _perform_dummy_operation(self, num_round: int) -> None:
        """Read, stash, and evict ``num_round`` random paths -- the padding that hides real round counts."""
        if num_round < 0:
            raise ValueError("The height is not enough, as the number of dummy operation required is negative.")

        for _ in range(num_round):
            leaf = self._get_new_leaf()

            self._client.add_read_path(label=self._name, leaves=[leaf])
            result = self._client.execute()
            path_data = result.require(self._name)
            path = self._decrypt_path_data(path=path_data)

            for bucket in path.values():
                for data in bucket:
                    self._stash.append(data)

            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")

            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
            self._client.execute()

    def _short_circuit_read(self, key: Any, num_round: int) -> bool:
        """Handle the dummy (``key is None``) / empty-tree case of a read-like op uniformly: do
        ``num_round`` dummy evictions and report that the caller should return ``None``. Routing every
        scheme through one place keeps the per-op obliviousness budget consistent (so a miss, a dummy,
        and a hit are indistinguishable in round count)."""
        if key is None or self.root is None:
            self._perform_dummy_operation(num_round=num_round)
            return True
        return False

    @abstractmethod
    def _op_round_bounds(self) -> dict[str, int]:
        """Per-op self-clearing round costs (retrieve the nodes, then evict them all back) keyed by
        'search'/'insert'/'delete', as a function of tree height. The non-distinguishable budget is the
        max over all of them (see ``_op_budget``)."""
        raise NotImplementedError

    def _op_budget(self, op: str) -> int:
        """Total rounds ``op`` pads to. ``distinguishable=True``: each op uses its own bound (so the op
        type leaks). Default (False): every op pads to the heaviest bound, hiding the op type."""
        bounds = self._op_round_bounds()
        if self._distinguishable:
            return bounds[op]
        return max(bounds.values())

    def _pad_to(self, budget: int) -> None:
        """Pad the current op to ``budget`` total rounds (real moves are tracked in _op_rounds)."""
        self._perform_dummy_operation(num_round=budget - self._op_rounds)

    @abstractmethod
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        """Build this scheme's in-memory tree from a non-empty ``data`` list; return its storage blocks
        and the (key, leaf) pointer to its root."""
        raise NotImplementedError

    def _new_ods_tree(self) -> BinaryTree:
        """Build the empty binary-tree storage every scheme fills (shared construction args)."""
        return BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._max_block_size,
            encryption=self._encryptor is not None,
        )

    @staticmethod
    def _normalize_pairs(data: KV_LIST) -> list[KVPair]:
        return [KVPair(key=pair[0], value=pair[1]) for pair in data]

    def _init_ods_storage(self, data: KV_LIST | None) -> BinaryTree:
        """Build the ODS-tree binary storage for the input key-value pairs."""
        tree = self._new_ods_tree()

        # Build the scheme's in-memory tree from the pairs, then flush its nodes to ORAM storage.
        if data:
            blocks, root = self._build_ods_blocks(data=self._normalize_pairs(data))
            for block in blocks:
                tree.fill_data_to_storage_leaf(data=block)
            self.root = root

        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    @override
    def init_server_storage(self, data: KV_LIST | None = None) -> None:
        self._client.init_storage(storage={self._name: self._init_ods_storage(data=data)})

    def _init_mul_tree_ods_storage(self, data_list: list[KV_LIST] | None) -> tuple[BinaryTree, list[ROOT | None]]:
        """Build one ODS tree per pair-list into shared storage; return it plus each tree's root."""
        tree = self._new_ods_tree()

        root_list: list[ROOT | None] = []
        for data in data_list or []:
            if data:
                blocks, root = self._build_ods_blocks(data=self._normalize_pairs(data))
                for block in blocks:
                    tree.fill_data_to_storage_leaf(data=block)
                root_list.append(root)
            else:
                root_list.append(None)

        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree, root_list

    def init_mul_tree_server_storage(self, data_list: list[KV_LIST] | None = None) -> list[ROOT | None]:
        """Store an ODS holding multiple trees; return the list of their roots."""
        tree, root_list = self._init_mul_tree_ods_storage(data_list=data_list)
        self._client.init_storage(storage={self._name: tree})
        return root_list
