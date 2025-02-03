# Towards Practical Oblivious Map

Xinle Cao, Weiqi Feng, Jian Liu, Jinjin Zhou, Wenjing Fang, Lei Wang, Quanqing Xu, Chuanhui Yang and Kui Ren.

(**Abstract**) Oblivious map is a crucial component in encrypted databases (EDBs), utilized to safeguard against the
server inferring sensitive information from client's encrypted **key-value pairs** based on access patterns. Despite its
widespread usage and importance, existing OMAP solutions face practical challenges, including the need for a large
number of interaction rounds between the client and server, as well as the substantial communication bandwidth
requirements. For example, the state-of-the-art protocol named ObliDB (VLDB 2020) still requires $O(\log{n})$
interaction rounds and $O(\log^2{n})$ communication bandwidth per access, where $n$ denotes the total number of
key-value pairs stored.

In this work, we present new OMAP constructions with the best theoretical complexity and actual performance up to now:
only $O(\log{n}/\log{\log{n}})$ interaction rounds and $O(\log^2{n}/\log{\log{n}})$ communication bandwidth are
required. Notably, as far as we know, they are the first **tree-based** OMAPs that overcome the $O(\log^{2}{n})$
communication bandwidth. The new advances result from a new framework for designing OMAPs: instead of using a **search
tree** in prior works, we propose new constructions based on **hash tables**. We implement both our proposed
constructions and prior methods. The experimental results demonstrate that our constructions substantially outperform
prior methods in terms of efficiency.

The full version of the paper is posted [here](https://eprint.iacr.org/2024/1650).

## ORAM Methods

- [Insecure Freecursive (ASPLOS 2015)](https://people.csail.mit.edu/devadas/pubs/freecursive.pdf), which is
  implemented [here](daoram/orams/freecursive_oram.py). The `reset_method` needs to be set to `"hard"`.
- [Secure Freecursive with probabilistic resets (TCC 2017)](https://eprint.iacr.org/2016/1084), which is
  implemented [here](daoram/orams/freecursive_oram.py). The `reset_method` needs to be set to `"prob"`.
- The proposed DAORAM with deterministic resets, which is implemented [here](daoram/orams/da_oram.py).

## OMAP Methods

- [OMAP based on AVL (CCS 2014)](https://dl.acm.org/doi/10.1145/2660267.2660314), which is
  implemented [here](daoram/omaps/avl_ods_omap.py). The `distinguishable_search` needs to be set to `False`.
- [OMAP based on optimized AVL (VLDB 2024)](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf), which is
  implemented [here](daoram/omaps/avl_ods_omap.py). The `distinguishable_search` needs to be set to `True`.
- [OMAP based on the B+ tree (VLDB 2020)](https://people.eecs.berkeley.edu/~matei/papers/2020/vldb_oblidb.pdf), which is
  implemented [here](daoram/omaps/bplus_ods_omap.py).
- The proposed OMAP framework is implemented [here](daoram/omaps/oram_ods_omap.py). It can be instantiated with any ORAM
  class
  contained in this repo combined with any OMAP class in this repo.

The open-sourced repositories we used as reference for the OMAP based on AVL tree and OMAP based on B+ tree are
here: [AVL](https://github.com/obliviousram/oblivious-avl-tree) [B+ tree](https://github.com/SabaEskandarian/ObliDB).

## Project Structure

- The [demo](demo) folder consists of demonstrations of how to use socket to set up ORAM/OMAP server and client.
- The [dependency](daoram/dependency) folder consists of some dependencies used including the socket, cryptography, etc.
- The [omaps](daoram/omaps) folder consists of all the OMAP constructions considered.
- The [orams](daoram/orams) folder consists of all the ORAM constructions considered.
- The [tests](tests) folder consists of test cases for validating correctness of our implementations.

## How to run this code

You need to first install the package listed in [`requirements.txt`](requirements.txt). If you want to run the schemes
with "local server", sample usages can be found in [`tests/test_orams.py`](tests/test_orams.py)
or [`tests/test_omaps.py`](tests/test_omaps.py). If you wish to set up a remote server, you should first
run [`demo/server.py`](demo/server.py) on the server and then run [`demo/oram_client.py`](demo/oram_client.py)
or [`demo/oram_client.py`](demo/omap_client.py) on your client device.
