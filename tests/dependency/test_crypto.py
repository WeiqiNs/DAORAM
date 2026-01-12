import os
import pickle

from daoram.dependency import AesGcm, Blake2Prf, FeistelPrp


class TestAesGcm:
    def test_aesgcm(self):
        # Create a new AesGcm instance.
        aesgcm = AesGcm()
        # Check the correctness of the scheme.
        assert aesgcm.dec(aesgcm.enc("Hello".encode())) == "Hello".encode()

    def test_aesgcm_with_pickle(self):
        # Create a new AesGcm instance.
        aesgcm = AesGcm()
        # Create some random data.
        data = [0, 1, [2, 3, 4, 5], os.urandom(100)]
        # Pickle dump some data.
        pickle_data = pickle.dumps(data)
        # Check the correctness of the scheme.
        assert aesgcm.dec(aesgcm.enc(pickle_data)) == pickle_data
        assert pickle.loads(aesgcm.dec(aesgcm.enc(pickle_data))) == data

    def test_ciphertext_length(self):
        # Create a new AesGcm instance.
        aesgcm = AesGcm()
        # Verify ciphertext length calculation.
        for length in [0, 1, 15, 16, 17, 32, 100]:
            plaintext = os.urandom(length)
            expected = aesgcm.ciphertext_length(length)
            actual = len(aesgcm.enc(plaintext))
            assert expected == actual

    def test_different_key_sizes(self):
        # Test with different key sizes.
        for key_size in [16, 24, 32]:
            aesgcm = AesGcm(key_byte_length=key_size)
            plaintext = b"Test data for encryption"
            assert aesgcm.dec(aesgcm.enc(plaintext)) == plaintext


class TestPrf:
    def test_prf(self):
        # Create a new prf instance.
        prf = Blake2Prf()
        # We need to make sure this algorithm is deterministic.
        assert prf.digest("Hello".encode()) == prf.digest("Hello".encode())


class TestPrp:
    def test_prp_permute_inverse(self):
        # Create a new PRP instance.
        prp = FeistelPrp(domain_size=100)
        # Check that inverse reverses permute for all values in domain.
        for x in range(100):
            y = prp.permute(x)
            assert prp.inverse(y) == x

    def test_prp_is_bijection(self):
        # Create a new PRP instance.
        prp = FeistelPrp(domain_size=50)
        # Check that permute produces all unique outputs (bijection).
        outputs = [prp.permute(x) for x in range(50)]
        assert len(set(outputs)) == 50
        # All outputs should be in [0, 50).
        assert all(0 <= y < 50 for y in outputs)

    def test_prp_deterministic(self):
        # Create a PRP with a fixed key.
        key = os.urandom(16)
        prp1 = FeistelPrp(domain_size=100, key=key)
        prp2 = FeistelPrp(domain_size=100, key=key)
        # Same key should produce same permutation.
        for x in range(100):
            assert prp1.permute(x) == prp2.permute(x)

    def test_prp_different_keys(self):
        # Different keys should produce different permutations.
        prp1 = FeistelPrp(domain_size=100)
        prp2 = FeistelPrp(domain_size=100)
        # Very unlikely that all outputs match with different keys.
        outputs1 = [prp1.permute(x) for x in range(100)]
        outputs2 = [prp2.permute(x) for x in range(100)]
        assert outputs1 != outputs2

    def test_prp_non_power_of_two(self):
        # Test with non-power-of-2 domain size (tests cycle-walking).
        prp = FeistelPrp(domain_size=37)
        outputs = [prp.permute(x) for x in range(37)]
        # Should still be a valid bijection.
        assert len(set(outputs)) == 37
        assert all(0 <= y < 37 for y in outputs)
