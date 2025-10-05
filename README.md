
# A Python library for important oblivious algorithms

In this library, we implment some classical oblivious algorithms including ORAM, OMAP, and other oblivious algorithms. Note all these algorithms aim to the client/server paradigm where the client-side execution is allowed to be non-oblivious.

We list our related papers as below for reference.

- [Towards Practical Oblivious Map](https://dl.acm.org/doi/10.14778/3712221.3712235)

## ORAM Methods

- [Insecure Freecursive (ASPLOS 2015)](https://people.csail.mit.edu/devadas/pubs/freecursive.pdf), which is
  implemented [here](daoram/oram/freecursive_oram.py). The `reset_method` needs to be set to `"hard"`.
- [Secure Freecursive with probabilistic resets (TCC 2017)](https://eprint.iacr.org/2016/1084), which is
  implemented [here](daoram/oram/freecursive_oram.py). The `reset_method` needs to be set to `"prob"`.
- The proposed DAORAM with deterministic resets, which is implemented [here](daoram/oram/da_oram.py).

## OMAP Methods

- [OMAP based on AVL (CCS 2014)](https://dl.acm.org/doi/10.1145/2660267.2660314), which is
  implemented [here](daoram/omap/avl_ods_omap.py). The `distinguishable_search` needs to be set to `False`.
- [OMAP based on optimized AVL (VLDB 2024)](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf), which is
  implemented [here](daoram/omap/avl_ods_omap.py). The `distinguishable_search` needs to be set to `True`.
- [OMAP based on the B+ tree (VLDB 2020)](https://people.eecs.berkeley.edu/~matei/papers/2020/vldb_oblidb.pdf), which is
  implemented [here](daoram/omap/bplus_ods_omap.py).
- The proposed OMAP framework is implemented [here](daoram/omap/oram_ods_omap.py). It can be instantiated with any ORAM
  class
  contained in this repo combined with any OMAP class in this repo.

The open-sourced repositories we used as reference for the OMAP based on AVL tree and OMAP based on B+ tree are
here: [AVL](https://github.com/obliviousram/oblivious-avl-tree) [B+ tree](https://github.com/SabaEskandarian/ObliDB).

## Project Structure

- The [demo](demo) folder consists of demonstrations of how to use socket to set up ORAM/OMAP server and client.
- The [dependency](daoram/dependency) folder consists of some dependencies used including the socket, cryptography, etc.
- The [omaps](daoram/omap) folder consists of all the OMAP constructions considered.
- The [orams](daoram/oram) folder consists of all the ORAM constructions considered.
- The [tests](tests) folder consists of test cases for validating correctness of our implementations.

## How to run this code

You need to first install the package listed in [`requirements.txt`](requirements.txt). If you want to run the schemes
with "local server," sample usages can be found in [`tests/test_orams.py`](tests/test_orams.py)
or [`tests/test_omaps.py`](tests/test_omaps.py). If you wish to set up a remote server, you should first
run [`demo/server.py`](demo/server.py) on the server and then run [`demo/oram_client.py`](demo/oram_client.py)
or [`demo/oram_client.py`](demo/omap_client.py) on your client device.
