import os
import pickle

from daoram.dependency import AesGcm, Blake2Prf


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
