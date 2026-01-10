import pytest

from daoram.dependency import InteractLocalServer
from daoram.omap.group_path_omap import GroupPathOmap


def test_group_path_omap_basic():

    # small example data set
    data = [(i, f"v{i}") for i in range(16)]

    server = InteractLocalServer()

    # create OMAP with 4 groups
    omap = GroupPathOmap(num_groups=pow(2,10), key_size=4, data_size=16, client=server, bucket_size=4,
                         use_encryption=True)

    # init server storage with provided data
    omap.init_server_storage(data=data)

    # search all keys and verify
    for k, v in data:
        got = omap.search(k)
        assert got == v

    for k, v in data:
        got = omap.search(k)
        assert got == v


def test_group_path_omap_insert_and_update():
    data = [(i, f"v{i}") for i in range(8)]
    server = InteractLocalServer()
    omap = GroupPathOmap(num_groups=pow(2,10), key_size=4, data_size=16, client=server, bucket_size=4,
                         use_encryption=False)
    omap.init_server_storage(data=data)

    # update an existing key
    omap.insert(3, "new3")
    assert omap.search(3) == "new3"

    # insert a new key (should be possible)
    omap.insert(100, "v100")
    assert omap.search(100) == "v100"

if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, "-v"])