import os
import pickle

import pytest
from cryptography.exceptions import InvalidTag

from oblivlib.dependency import AesGcm, Blake2Prf, FeistelPrp


class TestAesGcm:
    def test_round_trip(self):
        aesgcm = AesGcm()
        assert aesgcm.dec(aesgcm.enc(b"Hello")) == b"Hello"

    def test_round_trip_with_pickle(self):
        aesgcm = AesGcm()
        data = [0, 1, [2, 3, 4, 5], os.urandom(100)]
        pickle_data = pickle.dumps(data)
        assert aesgcm.dec(aesgcm.enc(pickle_data)) == pickle_data
        assert pickle.loads(aesgcm.dec(aesgcm.enc(pickle_data))) == data

    def test_ciphertext_length(self):
        aesgcm = AesGcm()
        for length in [0, 1, 15, 16, 17, 32, 100]:
            assert aesgcm.ciphertext_length(length) == len(aesgcm.enc(os.urandom(length)))

    def test_different_key_sizes(self):
        for key_size in [16, 24, 32]:
            aesgcm = AesGcm(key_byte_length=key_size)
            assert aesgcm.dec(aesgcm.enc(b"Test data for encryption")) == b"Test data for encryption"

    def test_invalid_key_size_raises(self):
        with pytest.raises(ValueError):
            AesGcm(key_byte_length=20)

    def test_tampered_ciphertext_raises(self):
        aesgcm = AesGcm()
        ciphertext = bytearray(aesgcm.enc(b"authentic message"))
        ciphertext[-1] ^= 0x01
        with pytest.raises(InvalidTag):
            aesgcm.dec(bytes(ciphertext))

    def test_distinct_nonce_per_encryption(self):
        aesgcm = AesGcm()
        assert aesgcm.enc(b"same") != aesgcm.enc(b"same")


class TestPrf:
    def test_deterministic(self):
        prf = Blake2Prf()
        assert prf.digest(b"Hello") == prf.digest(b"Hello")

    def test_same_key_same_digest(self):
        key = os.urandom(Blake2Prf.KEY_SIZE)
        assert Blake2Prf(key=key).digest(b"msg") == Blake2Prf(key=key).digest(b"msg")

    def test_digest_mod_n_in_range(self):
        prf = Blake2Prf()
        for i in range(100):
            assert 0 <= prf.digest_mod_n(message=str(i).encode(), mod=17) < 17


class TestPrp:
    def test_permute_inverse_round_trip(self):
        prp = FeistelPrp(domain_size=100)
        for x in range(100):
            assert prp.inverse(prp.permute(x)) == x

    def test_is_bijection(self):
        prp = FeistelPrp(domain_size=50)
        outputs = [prp.permute(x) for x in range(50)]
        assert sorted(outputs) == list(range(50))

    def test_deterministic_with_fixed_key(self):
        key = os.urandom(16)
        prp_a = FeistelPrp(domain_size=100, key=key)
        prp_b = FeistelPrp(domain_size=100, key=key)
        assert [prp_a.permute(x) for x in range(100)] == [prp_b.permute(x) for x in range(100)]

    def test_different_keys_differ(self):
        prp_a = FeistelPrp(domain_size=100)
        prp_b = FeistelPrp(domain_size=100)
        assert [prp_a.permute(x) for x in range(100)] != [prp_b.permute(x) for x in range(100)]

    def test_non_power_of_two_bijection_and_inverse(self):
        prp = FeistelPrp(domain_size=37)
        outputs = [prp.permute(x) for x in range(37)]
        assert sorted(outputs) == list(range(37))
        for x in range(37):
            assert prp.inverse(prp.permute(x)) == x

    def test_min_domain_size(self):
        prp = FeistelPrp(domain_size=2)
        assert sorted(prp.permute(x) for x in range(2)) == [0, 1]
        for x in range(2):
            assert prp.inverse(prp.permute(x)) == x

    def test_domain_too_small_raises(self):
        with pytest.raises(ValueError):
            FeistelPrp(domain_size=1)

    def test_out_of_range_raises(self):
        prp = FeistelPrp(domain_size=10)
        for bad in (-1, 10, 11):
            with pytest.raises(ValueError):
                prp.permute(bad)
            with pytest.raises(ValueError):
                prp.inverse(bad)
