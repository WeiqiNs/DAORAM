# Grove Code Audit

## Scope

This audit separates the Grove execution path from unrelated experimental
modules. A passing test means the current implementation satisfies the tested
contract; it is not a proof of the paper's probabilistic overflow bound.

## Module Boundaries

| Layer | Responsibility | Current evidence |
| --- | --- | --- |
| `daoram.dependency` | storage, trees, crypto, and batched server interaction | 67 tests pass in 0.80 s |
| `daoram.oram` | Path ORAM variants and multi-path reads/evictions | 48 tests pass in 149.72 s |
| `daoram.omap` | AVL/B+ OMAPs and cached/batched variants | 65 tests pass in 265.37 s |
| `daoram.graph.grove` | conference-version vertex-centered protocol | 127 graph tests pass as part of the graph suite |
| `daoram.graph.first_class_edge` | executable entity/reference/update state model | invariant and randomized state-machine tests pass |
| `daoram.graph.edge_oram` | concrete Edge Data ORAM storage | focused access/insert/delete tests pass |
| `daoram.graph.entity_grove_runtime` | synchronization between logical references and Edge Data ORAM | focused integration tests pass |
| `daoram.soram` | independent SORAM/SOMAP prototypes, not used by Grove | 2 pass and 16 fail after compatibility migration; algorithm/API redesign required |

## Confirmed Faults And Fixes

1. `MulPathOram.init_server_storage()` had two incompatible meanings. Normal
   ORAM use expects integer keys, while Grove metadata stores must start empty.
   Empty stores now pass `path_map={}` explicitly.
2. Grove tests initialized the PosMap metadata ORAM twice. `AVLOmapCached` now
   owns that initialization, and `Grove.init_server_storage()` owns the full
   initialization order.
3. The large random-graph test generated one-sided physical adjacency records.
   Grove's notification protocol requires records at both endpoints; logical
   direction is represented by endpoint roles.
4. `daoram.graph` lacked `__init__.py`, so `find_packages()` omitted all graph
   code from an installed distribution. The package now has an explicit API.
5. `grove_clean.py` was an indented, non-importable method fragment. It has
   been preserved under `docs/legacy/` and removed from the executable package.
6. SORAM used obsolete PRP, encryption-constructor, server-init, and list APIs.
   Those compatibility calls were migrated, exposing the remaining semantic
   mismatch rather than failing during import or setup.

## Open Risks

- One full-suite run observed `g_meta` stash size 229 exceeding a configured
  limit of 200 during 500 uniform lookups. A later full run and five isolated
  repetitions passed. Treat this as a stochastic capacity risk until a seeded
  leaf source and a repeated overflow-rate experiment quantify it.
- `grove.py` is still a large compatibility implementation. New journal work
  should stay in the state, storage, and runtime modules above; only stable
  mechanisms should later be integrated into the compatibility class.
- SORAM assumes dummy deletion and empty-tree search semantics no longer
  provided by OMAP. It needs an explicit dummy-record representation and a
  fresh protocol-level test oracle before further repair.

## Test Tiers

- Fast Grove development: `python -m pytest tests/dependency tests/graph -m "not slow" -q` (189 tests, about 25 s)
- Grove stress: `python -m pytest tests/graph -m slow -q`
- Storage regression: `python -m pytest tests/oram tests/omap -q`
- Known-independent debt: `python -m pytest tests/soram -q`

The fast tier protects interfaces and graph invariants. Stress and encrypted
storage runs remain separate because they dominate local validation time.
