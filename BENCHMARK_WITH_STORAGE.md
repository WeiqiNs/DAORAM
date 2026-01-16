# Bottom-Up SOMAP: Fast Benchmarking with Pre-built Storage

This guide explains how to accelerate Bottom-Up SOMAP benchmarks by pre-building server storage to disk. This is essential for large datasets (e.g., $N=2^{18}$ or larger) where network initialization takes hours.

## Workflow Overview

1.  **Generate Storage Files** (Once): Create `.pkl` files for the heavy Static ORAM ($D_S$) and lightweight Cache components ($O_W, O_R$).
2.  **Start Server**: Run the server in a separate terminal.
3.  **Run Client**: The client instructs the server to load these files instantly from disk.

---

## Step 1: Generate Storage Files

Use `scripts/prebuild_server.py`.

### Option A: Modular Generation (Recommended)

Generate the heavy $D_S$ once, and generate various cache configurations separately. This allows you to test different `cache_size` parameters without regenerating the massive dataset.

```bash
# 1. Generate the heavy Static ORAM base (e.g., N=16384, ~500MB)
# This file contains the main ORAM tree data.
python scripts/prebuild_server.py --num-data 16384 --value-size 4096 --output base_N14_DS.pkl --target ds_only

# 2. Generate cache components for different sizes (fast, ~seconds)
# These files contain O_W, O_R, Q_W, Q_R structures.
python scripts/prebuild_server.py --num-data 16384 --value-size 4096 --output cache_1023.pkl --cache-size 1023 --target cache_only
python scripts/prebuild_server.py --num-data 16384 --value-size 4096 --output cache_2047.pkl --cache-size 2047 --target cache_only
```

### Option B: Full Generation

Dump everything into one single file.
```bash
python scripts/prebuild_server.py --num-data 16384 --value-size 4096 --output full_N14.pkl --target full
```

---

## Step 2: Start Server

Open a terminal and run. Ensure the server has access to the `.pkl` files generated in Step 1.

```bash
python demo/server.py
```
*Keep this terminal running.*

---

## Step 3: Run Benchmark

In another terminal, use `scripts/benchmark_remote.py` with the `--load-storage` argument. You can specify multiple files separated by commas.

### Scenario 1: Testing with Cache Size = 1023

```bash
python scripts/benchmark_remote.py \
    --server-ip 127.0.0.1 \
    --num-data 16384 \
    --value-size 4096 \
    --protocol bottom_up \
    --ops 20 \
    --load-storage "base_N14_DS.pkl,cache_1023.pkl" \
    --cache-size 1023
```

### Scenario 2: Testing with Cache Size = 2047

Simply change the cache file and the `--cache-size` parameter. The server will merge the new cache structures over the existing $D_S$ in memory.

```bash
python scripts/benchmark_remote.py \
    --server-ip 127.0.0.1 \
    --num-data 16384 \
    --value-size 4096 \
    --protocol bottom_up \
    --ops 20 \
    --load-storage "base_N14_DS.pkl,cache_2047.pkl" \
    --cache-size 2047
```

### Scenario 3: Testing with Different B+ Tree Order

The `order` parameter controls the B+ tree branching factor in the cache ($O_W, O_R$). Since $D_S$ is order-independent, you can reuse the same base file.

**Generate:**
```bash
python scripts/prebuild_server.py --num-data 16384 --value-size 4096 --output cache_order8.pkl --order 8 --target cache_only
```

**Run:**
Ensure you pass the matching `--order` to the benchmark script so the client initializes its logic correctly.
```bash
python scripts/benchmark_remote.py \
    --server-ip 127.0.0.1 \
    --num-data 16384 \
    --value-size 4096 \
    --protocol bottom_up \
    --ops 20 \
    --load-storage "base_N14_DS.pkl,cache_order8.pkl" \
    --order 8
```

---

## Important Notes

1.  **Parameter Consistency**: 
    *   `num-data` and `value-size`: Must match the **Base** ($D_S$) file.
    *   `cache-size` and `order`: Must match the **Cache** file being loaded.
2.  **File Paths**: The storage files must be located where the **Server** can read them. If the server is on a different machine, you must upload the `.pkl` files to the server's directory first.
3.  **Client Hydration**: When the client loads storage from a file, it skips the `setup()` phase. The script automatically calls `restore_client_state()` to initialize local empty objects ($D_S, O_W, O_R$) so the client can perform operations immediately.
