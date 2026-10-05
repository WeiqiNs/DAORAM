import pytest

from oblivlib.dependency import PathOramConfig, RecursiveOramConfig
from oblivlib.oram import PathOram, RecursivePathOram


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_remote_client_server_round_trip(remote_client, encryptor, encrypted):
    oram = PathOram(
        PathOramConfig(num_data=256, data_size=10, client=remote_client, encryptor=encryptor if encrypted else None)
    )
    oram.init_server_storage()

    for i in range(256):
        oram.operate_on_key(key=i, value=b"%d" % i)

    for i in range(256):
        assert oram.operate_on_key(key=i) == b"%d" % i


def test_multiple_orams_share_one_client(client):
    a = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="a"))
    b = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="b"))
    a.init_server_storage()
    b.init_server_storage()

    for i in range(64):
        a.operate_on_key(key=i, value=b"a%d" % i)
        b.operate_on_key(key=i, value=b"b%d" % i)

    for i in range(64):
        assert a.operate_on_key(key=i) == b"a%d" % i
        assert b.operate_on_key(key=i) == b"b%d" % i


@pytest.mark.parametrize("mode", ["adopt", "attach", "stream"])
@pytest.mark.parametrize(
    "make_oram",
    [lambda **kw: PathOram(PathOramConfig(**kw)), lambda **kw: RecursivePathOram(RecursiveOramConfig(**kw))],
    ids=["path", "recursive"],
)
def test_build_file_handover_modes(make_oram, mode, tmp_path, encryptor, handover_client):
    client = handover_client(mode)
    build_file = tmp_path / "build.tree"
    oram = make_oram(num_data=128, data_size=10, client=client, encryptor=encryptor, build_file=build_file)
    oram.init_server_storage(data={key: b"%d" % (key * 3) for key in range(128)})

    assert not build_file.exists()
    for key in range(128):
        assert oram.operate_on_key(key=key, value=b"%d" % key) == b"%d" % (key * 3)
    for key in range(128):
        assert oram.operate_on_key(key=key) == b"%d" % key
    client.close()
