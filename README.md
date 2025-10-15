# A Python library for important oblivious algorithms

This library implements a series of classic oblivious algorithms, covering oblivious RAM (ORAM), oblivious map (OMAP), and algorithms for oblivious graph processing. It is built for client/server deployment scenarios in which the client is allowed to execute non-obliviously. 

We list the maintainer’s related publications, which motivated the development of this library, and then provide a detailed overview of all included algorithms.

- Enabling Index-free Adjacency in Oblivious Graph Processing with Delayed Duplications ([Full Version](Full_Version_Enabling_Index_free_Adjacency_in_Oblivious_Graph_Processing_with_Delayed_Duplications.pdf))
- [Towards Practical Oblivious Map (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235)

## Oblivious Graph Methods
- [GraphOS (VLDB 2024)](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf).
- [Grove with Delayed Duplications]().

## ORAM Methods

- [Path ORAM (CCS 2013)](https://dl.acm.org/doi/10.1145/2508859.2516660), implemented [here](daoram/oram/path_oram.py).
- [Recursive Path ORAM (ASIACRYPT 2011)](https://link.springer.com/chapter/10.1007/978-3-642-25385-0_11), implemented [here](daoram/oram/recursive_path_oram.py).
- [Insecure Freecursive ORAM (ASPLOS 2015)](https://people.csail.mit.edu/devadas/pubs/freecursive.pdf), implemented [here](daoram/oram/freecursive_oram.py). The `reset_method` should be set to `"hard"`.
- [Secure Freecursive ORAM with probabilistic resets (TCC 2017)](https://eprint.iacr.org/2016/1084), implemented [here](daoram/oram/freecursive_oram.py). The `reset_method` should be set to `"prob"`.
- [DAORAM with fixed reset and de-amortized cost (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235), implemented [here](daoram/oram/da_oram.py).

## OMAP Methods

- [OMAP based on AVL (CCS 2014)](https://dl.acm.org/doi/10.1145/2660267.2660314), implemented [here](daoram/omap/avl_ods_omap.py). Set `distinguishable_search` to `False`.
- [OMAP based on optimized AVL (VLDB 2024)](https://www.vldb.org/pvldb/vol16/p4324-chamani.pdf), implemented [here](daoram/omap/avl_ods_omap.py). Set `distinguishable_search` to `True`.
- [OMAP based on B+ tree (VLDB 2020)](https://people.eecs.berkeley.edu/~matei/papers/2020/vldb_oblidb.pdf), implemented [here](daoram/omap/bplus_ods_omap.py).
- [OMAP framework over an underlying search tree (VLDB 2025)](https://dl.acm.org/doi/10.14778/3712221.3712235), implemented [here](daoram/omap/oram_ods_omap.py). It can be instantiated with any ORAM class in this repo and combined with any tree-based OMAP class in this repo.
- Cache-optimized AVL and B+ tree variants are implemented [here](daoram/omap/avl_ods_omap_opt.py) and [here](daoram/omap/bplus_ods_omap_opt.py), respectively.

## Project Structure

- The [demo](demo) folder contains demonstrations showing how to use sockets to set up a server and client.  
- The [dependency](daoram/dependency) folder contains required dependencies, including socket and cryptography modules.  
- The [graph](daoram/graph) folder contains all oblivious graph constructions included in the library.  
- The [omap](daoram/omap) folder contains all OMAP constructions included in the library.  
- The [oram](daoram/oram) folder contains all ORAM constructions included in the library.  
- The [tests](tests) folder contains test cases for validating the correctness of the implementations.  

## How to run this code

1. Install the dependencies listed in [`requirements.txt`](requirements.txt).

2. Local server (for testing purposes):
   - See sample usages in [`tests/test_orams.py`](tests/test_orams.py) and [`tests/test_omaps.py`](tests/test_omaps.py).

3. Remote server:
   - Start the server on the remote machine, refer to: [`demo/server.py`](demo/server.py).
   - Run a client on your device, refer to: [`demo/oram_client.py`](demo/oram_client.py) or [`demo/omap_client.py`](demo/omap_client.py).

