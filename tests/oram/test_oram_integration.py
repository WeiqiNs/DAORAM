import pytest

from oblivlib.dependency import PathOramConfig
from oblivlib.oram import PathOram


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_remote_client_server_round_trip(remote_client, encryptor, encrypted):
    oram = PathOram(
        PathOramConfig(num_data=256, data_size=10, client=remote_client, encryptor=encryptor if encrypted else None)
    )
    oram.init_server_storage()

    for i in range(256):
        oram.operate_on_key(key=i, value=i)

    for i in range(256):
        assert oram.operate_on_key(key=i) == i


def test_multiple_orams_share_one_client(client):
    a = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="a"))
    b = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="b"))
    a.init_server_storage()
    b.init_server_storage()

    for i in range(64):
        a.operate_on_key(key=i, value=i)
        b.operate_on_key(key=i, value=i * 10)

    for i in range(64):
        assert a.operate_on_key(key=i) == i
        assert b.operate_on_key(key=i) == i * 10
