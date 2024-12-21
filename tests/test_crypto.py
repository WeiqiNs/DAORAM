import os
import pickle

import pytest

from daoram.dependency import Aes, Prf, pad_pickle, unpad_pickle


def test_pad_pickle():
    empty_str = b""
    assert unpad_pickle(pad_pickle(data=empty_str, length=100)) == empty_str

    data = b"data"
    assert unpad_pickle(pad_pickle(data=data, length=100)) == data

    with pytest.raises(ValueError, match="Padding is broken because of trailing null byte."):
        pad_pickle(data=b"\x00", length=100)


def test_aes():
    # Create a new aes instance.
    aes = Aes()
    # Check correctness of the scheme.
    assert aes.dec(aes.enc("Hello".encode())) == "Hello".encode()


def test_aes_with_pickle():
    # Create a new aes instance.
    aes = Aes()
    # Create some random data.
    data = [0, 1, [2, 3, 4, 5], os.urandom(100)]
    # Pickle dump some data.
    pickle_data = pickle.dumps(data)
    # Check correctness of the scheme.
    assert aes.dec(aes.enc(pickle_data)) == pickle_data
    assert pickle.loads(aes.dec(aes.enc(pickle_data))) == data


def test_prf():
    # Create a new prf instance.
    prf = Prf()
    # We need to make sure this algorithm is deterministic.
    assert prf.digest("Hello".encode()) == prf.digest("Hello".encode())
