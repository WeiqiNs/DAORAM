"""Behavioral suite run against every ODS OMAP via the ``omap_spec`` fixture (see conftest.py).

Three reinforcing kinds of check:
  * model oracle      -- random op sequences must agree with a plain ``dict`` at every step;
  * structural invariants -- the reconstructed tree must stay a valid, balanced, correctly-linked AVL;
  * access-pattern uniformity -- search/insert must touch a fixed number of paths regardless of key
    (the obliviousness property tests otherwise cannot see). Cached variants opt out.
"""

import math
import random

import pytest

from oblivlib.dependency import InteractLocalServer
from oblivlib.omap import AVLOmap, AVLOmapCached, BPlusOmap, BPlusOmapCached, GroupOmap, OramOstOmap
from oblivlib.omap.base_omap import BaseOmap


def test_all_schemes_implement_base_omap():
    for scheme in (AVLOmap, AVLOmapCached, BPlusOmap, BPlusOmapCached, GroupOmap, OramOstOmap):
        assert issubclass(scheme, BaseOmap)


class TestOmapBehavior:
    def test_round_trip(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=i, value=i)
        for i in range(200):
            assert omap.search(key=i) == i

    def test_search_empty(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        assert omap.search(key=5) is None
        if omap_spec.supports_delete:
            assert omap.delete(key=5) is None

    def test_dummy_ops(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        for i in range(20):
            omap.insert(key=i, value=i)
        assert omap.search(key=None) is None
        omap.insert(key=None, value=123)
        if omap_spec.supports_delete:
            assert omap.delete(key=None) is None
        for i in range(20):
            assert omap.search(key=i) == i

    @pytest.mark.parametrize("num_data", [1, 2, 7])
    def test_edge_sizes(self, omap_spec, num_data):
        omap = omap_spec.make(client=InteractLocalServer(), num_data=num_data)
        omap.init_server_storage()
        for i in range(num_data):
            omap.insert(key=i, value=i * 10)
        for i in range(num_data):
            assert omap.search(key=i) == i * 10

    def test_string_keys(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=128)
        omap.init_server_storage()
        for i in range(100):
            omap.insert(key=f"k{i:04d}", value=f"v{i}")
        for i in range(100):
            assert omap.search(key=f"k{i:04d}") == f"v{i}"
        assert omap.search(key="absent") is None

    def test_update_and_missing(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=128)
        omap.init_server_storage()
        for i in range(50):
            omap.insert(key=i, value=i)
        assert omap.search(key=10, value=999) == 10
        assert omap.search(key=10) == 999
        assert omap.search(key=10_000) is None

    def test_encryption_round_trip(self, omap_spec, client, encryptor):
        omap = omap_spec.make(client=client, num_data=128, encryptor=encryptor)
        omap.init_server_storage()
        for i in range(60):
            omap.insert(key=i, value=i * 3)
        for i in range(60):
            assert omap.search(key=i) == i * 3

    def test_file_backend_round_trip(self, omap_spec, client, test_file, encryptor):
        omap = omap_spec.make(client=client, num_data=128, filename=str(test_file), encryptor=encryptor)
        omap.init_server_storage()
        for i in range(60):
            omap.insert(key=i, value=i * 5)
        for i in range(60):
            assert omap.search(key=i) == i * 5

    def test_init_with_data(self, omap_spec, client):
        omap = omap_spec.make(client=client, num_data=256)
        omap.init_server_storage(data=[(f"{i}", f"{i}") for i in range(128)])
        for i in range(128, 200):
            omap.insert(key=f"{i}", value=f"{i}")
        for i in range(200):
            assert omap.search(key=f"{i}") == f"{i}"

    def test_mul_tree_init(self, omap_spec, num_data, client):
        extra = 3
        size_group = math.floor(math.log2(num_data))
        num_group = num_data // size_group
        init_data = [[(j, j) for j in range(i * 2 * size_group, (i * 2 + 1) * size_group)] for i in range(num_group)]

        omap = omap_spec.make(client=client, num_data=num_data)
        roots = omap.init_mul_tree_server_storage(data_list=init_data)

        for i, root in enumerate(roots):
            omap.root = root
            for j in range(extra):
                omap.insert(key=(i * 2 + 1) * size_group + j, value=(i * 2 + 1) * size_group + j)
            roots[i] = omap.root

        for i, root in enumerate(roots):
            omap.root = root
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.search(key=j) == j

    def test_model_oracle(self, omap_spec):
        rng = random.Random(1234)
        omap = omap_spec.make(client=InteractLocalServer(), num_data=128)
        omap.init_server_storage()
        read = omap.search
        model, keyspace = {}, list(range(128))
        for _ in range(800):
            key = rng.choice(keyspace)
            if rng.random() < 0.5:
                if key not in model:  # AVL build does not handle duplicate keys
                    value = rng.randint(0, 10**6)
                    omap.insert(key=key, value=value)
                    model[key] = value
            else:
                assert read(key=key) == model.get(key)
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)

    def test_invariants(self, omap_spec):
        rng = random.Random(99)
        omap = omap_spec.make(client=InteractLocalServer(), num_data=128)
        omap.init_server_storage()
        model = {}
        for _ in range(300):
            key = rng.randrange(128)
            if key not in model:
                omap.insert(key=key, value=key)
                model[key] = key
        if omap_spec.cached:
            omap._flush_local_to_stash()
        omap_spec.invariant_checker(omap, model)

    def test_delete_oracle(self, omap_spec):
        if not omap_spec.supports_delete:
            pytest.skip("scheme has no delete")
        rng = random.Random(2024)
        omap = omap_spec.make(client=InteractLocalServer(), num_data=128)
        omap.init_server_storage()
        model, keyspace = {}, list(range(128))
        for _ in range(900):
            key = rng.choice(keyspace)
            roll = rng.random()
            if roll < 0.45:
                if key not in model:
                    value = rng.randint(0, 10**6)
                    omap.insert(key=key, value=value)
                    model[key] = value
            elif roll < 0.75:
                assert omap.search(key=key) == model.get(key)
            elif model:  # delete needs a non-empty tree; exercise both present and absent keys
                assert omap.delete(key=key) == model.get(key)
                model.pop(key, None)
        if omap_spec.cached:
            omap._flush_local_to_stash()
        for key in keyspace:
            assert omap.search(key=key) == model.get(key)
        if omap_spec.cached:
            omap._flush_local_to_stash()
        omap_spec.invariant_checker(omap, model)

    def test_delete_empties_tree(self, omap_spec, client):
        if not omap_spec.supports_delete:
            pytest.skip("scheme has no delete")
        omap = omap_spec.make(client=client, num_data=64)
        omap.init_server_storage()
        for i in range(40):
            omap.insert(key=i, value=i)
        for i in range(40):
            assert omap.delete(key=i) == i
        assert omap.root is None


class TestOmapObliviousness:
    """Per-op access patterns must be key-independent: a fixed number of path rounds regardless of
    the key, a miss must look like a hit, and a dummy op(None) must look like a real op."""

    def test_search_access_uniform(self, oblivious_omap_spec, counting_server_cls):
        omap_spec = oblivious_omap_spec
        server = counting_server_cls()
        omap = omap_spec.make(client=server, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=i, value=i)
        hit = {server.rounds(omap.search, k)[0] for k in random.Random(0).sample(range(200), 40)}
        miss = {server.rounds(omap.search, 10_000 + j)[0] for j in range(10)}
        assert len(hit) == 1, f"search rounds vary by key: {sorted(hit)}"
        assert miss == hit, f"missing-key search distinguishable from hit: miss={miss} hit={hit}"
        if omap_spec.search_none:
            dummy = {server.rounds(omap.search, None)[0] for _ in range(5)}
            assert dummy == hit, f"search(None) distinguishable from a real search: {dummy} vs {hit}"

    def test_insert_access_uniform(self, oblivious_omap_spec, counting_server_cls):
        omap_spec = oblivious_omap_spec
        server = counting_server_cls()
        omap = omap_spec.make(client=server, num_data=256)
        omap.init_server_storage()
        rounds = {server.rounds(omap.insert, i, i)[0] for i in range(200)}
        assert len(rounds) == 1, f"insert rounds vary: {sorted(rounds)}"
        dummy = {server.rounds(omap.insert, None)[0] for _ in range(5)}
        assert dummy == rounds, f"insert(None) distinguishable from a real insert: {dummy} vs {rounds}"

    def test_op_type_hidden_when_not_distinguishable(self, oblivious_omap_spec, counting_server_cls):
        omap_spec = oblivious_omap_spec
        server = counting_server_cls()
        omap = omap_spec.make(client=server, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=i, value=i)
        insert_rounds = server.rounds(omap.insert, 10_000, 1)[0]
        search_rounds = server.rounds(omap.search, 50)[0]
        delete_rounds = server.rounds(omap.delete, 50)[0]
        assert insert_rounds == search_rounds == delete_rounds, (
            f"op type leaks: insert={insert_rounds} search={search_rounds} delete={delete_rounds}"
        )

    def test_distinguishable_mode_separates_ops(self, oblivious_omap_spec, counting_server_cls):
        omap_spec = oblivious_omap_spec
        server = counting_server_cls()
        omap = omap_spec.make(client=server, num_data=256, distinguishable=True)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=i, value=i)
        search_rounds = server.rounds(omap.search, 50)[0]
        delete_rounds = server.rounds(omap.delete, 50)[0]
        assert search_rounds < delete_rounds, f"search {search_rounds} should be < delete {delete_rounds}"

    def test_delete_access_uniform(self, delete_oblivious_omap_spec, counting_server_cls):
        omap_spec = delete_oblivious_omap_spec
        server = counting_server_cls()
        omap = omap_spec.make(client=server, num_data=256)
        omap.init_server_storage()
        for i in range(200):
            omap.insert(key=i, value=i)
        existing = {server.rounds(omap.delete, k)[0] for k in random.Random(3).sample(range(200), 30)}
        missing = {server.rounds(omap.delete, 10_000 + j)[0] for j in range(10)}
        dummy = {server.rounds(omap.delete, None)[0] for _ in range(5)}
        assert len(existing) == 1, f"delete(existing) rounds vary: {sorted(existing)}"
        assert missing == existing, f"delete(missing) distinguishable: miss={missing} hit={existing}"
        assert dummy == existing, f"delete(None) distinguishable: dummy={dummy} hit={existing}"
