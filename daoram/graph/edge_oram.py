"""Edge Data ORAM accessed only through endpoint-owned path references."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional

from daoram.dependency import Data, Encryptor, InteractServer
from daoram.graph.first_class_edge import EdgeRecord
from daoram.oram.mul_path_oram import MulPathOram


@dataclass(frozen=True)
class EdgeReference:
    """The information an endpoint adjacency slot needs to locate an edge."""

    edge_id: Any
    path: int


class EdgeDataOram:
    """A position-map-free interface over a multi-path ORAM.

    ``MulPathOram`` contains a conventional position map for its general API.
    This wrapper uses that map only while constructing initial storage and then
    clears it.  Every online operation must provide an ``EdgeReference`` that
    came from an endpoint vertex.
    """

    def __init__(
        self,
        num_edges: int,
        data_size: int,
        client: InteractServer,
        name: str = "edge",
        filename: Optional[str] = None,
        bucket_size: int = 4,
        stash_scale: int = 7,
        encryptor: Optional[Encryptor] = None,
    ) -> None:
        self._name = name
        self._client = client
        self._oram = MulPathOram(
            name=name,
            client=client,
            num_data=num_edges,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=filename,
        )

    @property
    def stash_size(self) -> int:
        return self._oram.stash_size

    @property
    def position_map_size(self) -> int:
        """Always zero after initialization in this endpoint-owned design."""
        return len(self._oram._pos_map)

    @property
    def leaf_range(self) -> int:
        return self._oram._leaf_range

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return current Edge Data ORAM storage and transport metrics."""
        label_read, label_written = self._client.get_label_bandwidth().get(
            self._name, (0, 0)
        )
        return {
            "label": self._name,
            "bytes_read": label_read,
            "bytes_written": label_written,
            "stash_size": self.stash_size,
            "max_stash_size": self._oram.max_stash,
            "position_map_size": self.position_map_size,
        }

    def init_server_storage(self, records: Iterable[EdgeRecord] = ()) -> None:
        """Initialize the Edge Data ORAM from first-class edge records."""
        record_list = list(records)
        data_map = {record.edge_id: record for record in record_list}
        path_map = {record.edge_id: record.path for record in record_list}
        if len(data_map) != len(record_list):
            raise ValueError("Duplicate edge ids in initial storage.")
        self._oram.init_server_storage(data_map=data_map, path_map=path_map)
        self._oram._pos_map.clear()

    def _read_paths(self, paths: list[int]) -> None:
        self._oram.queue_read(leaves=paths)
        result = self._oram.client.execute()
        self._oram.process_read_result(result)

    def _find_stash_block(self, edge_id: Any) -> Data:
        for block in self._oram.stash:
            if block.key == edge_id:
                return block
        raise KeyError(f"Edge {edge_id!r} was not found on the supplied path.")

    def _write_paths(self, paths: list[int]) -> None:
        self._oram.queue_write(leaves=paths)
        self._oram.client.execute()

    def access(
        self,
        references: Iterable[EdgeReference],
        value_updates: Optional[Mapping[Any, Any]] = None,
        new_paths: Optional[Mapping[Any, int]] = None,
    ) -> Dict[Any, EdgeRecord]:
        """Read and remap edges using only endpoint-supplied old paths."""
        refs = list(references)
        if not refs:
            return {}
        if len({ref.edge_id for ref in refs}) != len(refs):
            raise ValueError("The same edge cannot be accessed twice in one batch.")

        old_paths = [ref.path for ref in refs]
        self._read_paths(old_paths)

        result: Dict[Any, EdgeRecord] = {}
        for ref in refs:
            block = self._find_stash_block(ref.edge_id)
            record = block.value
            if not isinstance(record, EdgeRecord):
                raise TypeError("Edge Data ORAM contains a non-edge record.")

            new_path = (
                new_paths[ref.edge_id]
                if new_paths is not None and ref.edge_id in new_paths
                else self._oram._get_new_leaf()
            )
            new_value = (
                value_updates[ref.edge_id]
                if value_updates is not None and ref.edge_id in value_updates
                else record.value
            )
            remapped = replace(record, value=new_value, path=new_path)
            block.leaf = new_path
            block.value = remapped
            result[ref.edge_id] = remapped

        self._write_paths(old_paths)
        return result

    def insert(self, record: EdgeRecord, eviction_path: Optional[int] = None) -> None:
        """Insert one edge while reading and rewriting a public eviction path."""
        if any(block.key == record.edge_id for block in self._oram.stash):
            raise ValueError(f"Duplicate edge id: {record.edge_id!r}")
        path = self._oram._get_new_leaf() if eviction_path is None else eviction_path
        self._read_paths([path])
        if any(block.key == record.edge_id for block in self._oram.stash):
            raise ValueError(f"Duplicate edge id: {record.edge_id!r}")
        self._oram.stash.append(
            Data(key=record.edge_id, leaf=record.path, value=record)
        )
        self._write_paths([path])

    def delete(self, reference: EdgeReference) -> EdgeRecord:
        """Remove one edge using its endpoint-owned path reference."""
        self._read_paths([reference.path])
        block = self._find_stash_block(reference.edge_id)
        record = block.value
        self._oram.stash.remove(block)
        self._write_paths([reference.path])
        if not isinstance(record, EdgeRecord):
            raise TypeError("Edge Data ORAM contains a non-edge record.")
        return record
