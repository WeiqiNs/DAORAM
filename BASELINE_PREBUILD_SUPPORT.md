# Baseline B+ OMAP Pre-build & Restore Support

Support has been added for pre-building `BPlusOdsOmap` (Baseline) into a server storage file and restoring it during benchmark execution. This complements the existing support for Bottom-Up and Top-Down SOMAP protocols.

## Status: Completed

## Key Changes
1. **Server Pre-build Script (`scripts/prebuild_server.py`)**:
   - Added support for `--protocol baseline`.
   - **Metadata Injection**: Now automatically extracts `proto.root` (Root ID, Leaf) and saves it into the storage file under the key `{name}_root` (e.g., `baseline_root`). This solves the "headless client" problem where loading the tree structure was useless without the root pointer.

2. **Client Restoration Logic (`daoram/omap/bplus_ods_omap.py`)**:
   - Added `restore_client_state()` method.
   - It queries the server for `{name}_root` metadata after loading storage.
   - It updates `self._root` with the remote value, reconnecting the client to the B+ Tree.

3. **Benchmark Runner (`scripts/benchmark_remote.py`)**:
   - Updated `run_baseline` to accept `--load-storage` argument.
   - Uses `client.load_storage()` followed by `omap.restore_client_state()`.

## Usage
### 1. Pre-build the Server Storage
Generate the storage file containing the B+ Tree and the Root metadata.
```bash
python scripts/prebuild_server.py \
    --protocol baseline \
    --num-data 1000 \
    --value-size 64 \
    --order 4 \
    --output baseline_1k.pkl
```

### 2. Start the Server
Start the server (empty init).
```bash
python demo/server.py --port 10000
```
*(Note: The server does not load the file on startup. It waits for the client to send the load command.)*

### 3. Run the Benchmark with `--load-storage`
Run the client benchmark pointing to the file.
```bash
python scripts/benchmark_remote.py \
    --protocol baseline \
    --num-data 1000 \
    --value-size 64 \
    --order 4 \
    --server-ip localhost \
    --port 10000 \
    --load-storage baseline_1k.pkl
```
The client will:
1. Connect to the server.
2. Instruct the server to load `baseline_1k.pkl`.
3. Fetch the Root pointer from the loaded metadata.
4. Run the operations immediately (skipping the slow tree build).
