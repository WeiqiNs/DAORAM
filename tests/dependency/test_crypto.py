import os
import pickle

from daoram.dependency import Aes, Prf


class TestCrypto:
    def test_aes(self):
        # Create a new aes instance.
        aes = Aes()
        # Check the correctness of the scheme.
        assert aes.dec(aes.enc("Hello".encode())) == "Hello".encode()

    def test_aes_with_pickle(self):
        # Create a new aes instance.
        aes = Aes()
        # Create some random data.
        data = [0, 1, [2, 3, 4, 5], os.urandom(100)]
        # Pickle dump some data.
        pickle_data = pickle.dumps(data)
        # Check the correctness of the scheme.
        assert aes.dec(aes.enc(pickle_data)) == pickle_data
        assert pickle.loads(aes.dec(aes.enc(pickle_data))) == data

    def test_prf(self):
        # Create a new prf instance.
        prf = Prf()
        # We need to make sure this algorithm is deterministic.
        assert prf.digest("Hello".encode()) == prf.digest("Hello".encode())

