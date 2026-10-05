import shutil
from pathlib import Path

import pytest

from oblivlib.dependency import AesGcm, Data
from oblivlib.dependency.codec import DefaultCodec
from oblivlib.dependency.heap_index import tree_size
from oblivlib.dependency.path_cipher import PathCipher, make_path_cipher
from oblivlib.dependency.tree_builder import FileImage, MemoryImage, TreeImage, build_tree

_CODEC = DefaultCodec(max_block_bytes=64)


def _rows(image: TreeImage) -> list[bytes]:
    if isinstance(image, MemoryImage):
        return image.rows
    data = image.path.read_bytes()
    assert len(data) == tree_size(image.level) * image.row_bytes
    return [data[i * image.row_bytes : (i + 1) * image.row_bytes] for i in range(tree_size(image.level))]


def _keys(cipher: PathCipher, image: TreeImage) -> list[list[int]]:
    return [[data.key for data in cipher.open_bucket(row)] for row in _rows(image)]


@pytest.mark.parametrize("mode", ["memory_plain", "memory_sealed", "file_sealed"])
def test_build_places_blocks_deepest_first_and_reports_overflow(mode, tmp_path):
    cipher = make_path_cipher(_CODEC, None if mode == "memory_plain" else AesGcm(), bucket_size=1)
    blocks = [Data(key=key, leaf=0, value=b"v") for key in range(5)] + [Data(key=9, leaf=5, value=b"w")]
    build_file = tmp_path / "build.tree" if mode == "file_sealed" else None

    result = build_tree(blocks, level=4, bucket_size=1, cipher=cipher, build_file=build_file)

    keys = _keys(cipher, result.image)
    assert [keys[7], keys[3], keys[1], keys[0], keys[12]] == [[0], [1], [2], [3], [9]]
    assert sum(len(bucket) for bucket in keys) == 5
    assert [data.key for data in result.overflow] == [4]
    if cipher.row_bytes is not None:
        assert {len(row) for row in _rows(result.image)} == {cipher.row_bytes}


def test_three_handovers_produce_identical_server_state(tmp_path, handover_client):
    cipher = make_path_cipher(_CODEC, AesGcm(), bucket_size=2)
    blocks = [Data(key=key, leaf=key % 8, value=bytes([key]) * 4) for key in range(12)]
    image = build_tree(blocks, level=4, bucket_size=2, cipher=cipher, build_file=tmp_path / "build.tree").image
    assert isinstance(image, FileImage)
    expected = _rows(image)

    sources = tmp_path / "sources"
    sources.mkdir()
    for mode in ("adopt", "attach", "stream"):
        client = handover_client(mode)
        source = Path(shutil.copy(image.path, sources / f"{mode}.tree"))
        client.host_tree("t", FileImage(level=image.level, row_bytes=image.row_bytes, path=source))
        assert not source.exists()
        assert (client.metrics.payload_bytes > 0) is (mode == "stream")

        client.add_read_path("t", range(8))
        rows = client.execute().require("t")
        assert [rows[index] for index in range(tree_size(image.level))] == expected
        client.close()
