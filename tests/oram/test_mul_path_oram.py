import random

from oblivlib.dependency import UNSET, MulPathOramConfig
from oblivlib.oram import MulPathOram


class TestMulPathOram:
    def test_batch_write(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        updates = {0: 100, 1: 101, 2: 102}
        old_values = oram.operate_on_keys(key_value_map=updates)

        assert old_values[0] == 0
        assert old_values[1] == 1
        assert old_values[2] == 2

        for key, new_value in updates.items():
            assert oram.operate_on_key(key=key) == new_value

    def test_batch_read_only(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i * 2)

        read_map = {0: UNSET, 5: UNSET, 10: UNSET}
        results = oram.operate_on_keys(key_value_map=read_map)

        assert results[0] == 0
        assert results[5] == 10
        assert results[10] == 20

        assert oram.operate_on_key(key=0) == 0
        assert oram.operate_on_key(key=5) == 10
        assert oram.operate_on_key(key=10) == 20

    def test_with_enc(self, num_data, client, encryptor):
        oram = MulPathOram(
            MulPathOramConfig(
                num_data=num_data, data_size=10, client=client, encryptor=encryptor, stash_scale_multiplier=5
            )
        )
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        keys_to_read = [0, 1, 2, 3, 4]
        results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys_to_read})

        for key in keys_to_read:
            assert results[key] == key

        updates = {5: 50, 6: 60}
        oram.operate_on_keys(key_value_map=updates)

        assert oram.operate_on_key(key=5) == 50
        assert oram.operate_on_key(key=6) == 60

    def test_with_file(self, num_data, client, test_file):
        oram = MulPathOram(
            MulPathOramConfig(
                num_data=num_data, data_size=10, client=client, filename=str(test_file), stash_scale_multiplier=3
            )
        )
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        results = oram.operate_on_keys(key_value_map={0: UNSET, 1: UNSET, 2: UNSET})
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

    def test_random_batch_operations(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=10))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        for _ in range(10):
            batch_size = random.randint(2, min(10, num_data))
            keys = random.sample(range(num_data), batch_size)

            results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})

            assert len(results) == batch_size
            for key in keys:
                assert results[key] == key

    def test_without_eviction(self, num_data, client):
        oram = MulPathOram(MulPathOramConfig(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3))
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        keys_to_read = [0, 1, 2]
        results = oram.operate_on_keys_without_eviction(key_value_map={k: UNSET for k in keys_to_read})

        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

        oram.eviction_for_mul_keys(updates={0: 100, 1: 101})

        assert oram.operate_on_key(key=0) == 100
        assert oram.operate_on_key(key=1) == 101
        assert oram.operate_on_key(key=2) == 2
