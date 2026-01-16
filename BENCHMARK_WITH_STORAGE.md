# 使用 pre-built storage 加速 Benchmark

DAORAM 支持预先生成服务器端的存储状态文件（Tree, OMAP, Queues 等），并在 Benchmark 时直接加载，从而跳过耗时的初始化过程。目前 Bottom-Up 和 Top-Down 协议均已支持此功能。

## 1. 生成存储文件 (Server-side)

使用 `scripts/prebuild_server.py` 生成存储文件。该脚本会在本地运行一个协议实例，完成 setup 过程，并将结果 `pickle` 到磁盘。

### 参数说明
- `--protocol`: 选择协议 ( `bottom_up` 或 `top_down` )。
- `--num-data (N)`: 数据总项数。
- `--cache-size (C)`: 缓存大小。
- `--target`: 指定保存内容。
    - `full`: 保存所有内容 (Tree, DB/DS, OMAPs, Queues)。**推荐用于完整测试**。
    - `ds_only`: 仅保存主要存储 (Tree 或 DS)。用于测试不同 Cache 大小搭配同一 Tree。对于 Top-Down，这包括 `Tree` 和 `DB`。
    - `cache_only`: 仅保存 Cache 部分 (OMAPs)。
- `--output`: 输出文件名。

### 示例

**生成 Top-Down 协议的完整存储 (N=65536, C=1024):**
```bash
python scripts/prebuild_server.py \
    --protocol top_down \
    --num-data 65536 \
    --cache-size 1024 \
    --target full \
    --output server_td_N65k_C1k.pkl
```

**生成 Bottom-Up 协议的完整存储:**
```bash
python scripts/prebuild_server.py \
    --protocol bottom_up \
    --num-data 65536 \
    --cache-size 1024 \
    --target full \
    --output server_bu_N65k_C1k.pkl
```

## 2. 运行 Benchmark (Client-side)

使用 `scripts/benchmark_remote.py` 并配合 `--load-storage` 参数。

### 步骤
1. **启动服务器** (Server):
   ```bash
   python demo/server.py --port 10000
   ```

2. **运行客户端** (Client):
   ```bash
   # 注意：
   # 1. 必须使用与生成时相同的 protocol, num-data, cache-size 等参数。
   # 2. 如果使用 target=full 的文件，cache-size 必须完全一致。
   
   python scripts/benchmark_remote.py \
       --protocol top_down \
       --server-ip localhost \
       --port 10000 \
       --num-data 65536 \
       --cache-size 1024 \
       --load-storage server_td_N65k_C1k.pkl
   ```

## 3. Top-Down 协议注意事项

Top-Down 协议依赖 PRF 进行 Addressing，因此 Client 需要与生成文件的 Server 拥有相同的密钥种子。
- 目前代码中默认使用全 `0` 的 AES Key (或 Mock Key) 来保证这一确定性。
- 如果加载了 `full` 文件，Client 会自动检测并**跳过**耗时的缓存重置（Empty initialization），直接使用文件中的状态。
- 如果仅加载 `ds_only` (Tree/DB)，Client 会在连接建立后，向 Server 发送命令以初始化空的 OMAPs (Or, Ow, Ob) 和 Queues。

## 4. 常见问题 (FAQ)

**Q: 是否可以只加载 Tree，然后用不同的 Cache Size 测试？**
A: 可以。
1. 生成时使用 `--target ds_only` 保存 Tree/DS。
2. Benchmark 时指定 `--load-storage your_tree.pkl`，并设置所需的 `--cache-size`。
3. Client 会检测到需要 Reset Cache，并自动初始化新的 Cache 结构。

**Q: 现在的初始化是不是都可以从文件里直接读取 Tree, Ds 甚至 Or 和 Ow？**
A: **是的**。如果使用 `--target full` 生成文件，它包含了协议运行所需的所有服务器端状态：
- **Tree**: 二叉树结构 (Top-Down) 或 路径 (Bottom-Up)。
- **Ds/DB**: 主存储区域。
- **Or/Ow**: 读写 OMAP 结构。
- **Ob**: Backup OMAP (Top-Down)。
- **Qr/Qw**: 请求队列。
加载后 Client 几乎瞬间即可进入可用状态，无需重新上传任何数据。

**Q: 能够根据参数比如 cache size 读取不同的 Ow, Or 吗？**
A: **可以**。这种 "Mix and Match" 的能力正是设计 `ds_only` 和 `cache_only` 的初衷。
1. **生成阶段**：您可以生成一个通用的 Base (Tree/DB)，然后为不同的 Cache Size 生成多个 Cache 文件。
   ```bash
   # 1. 生成 Base (Tree/DB), 耗时较长
   python scripts/prebuild_server.py --protocol top_down --num-data 65536 --target ds_only --output base_td_N65k.pkl
   
   # 2. 生成不同大小的 Cache (Ow/Or/Ob/Queues), 耗时极短
   python scripts/prebuild_server.py --protocol top_down --num-data 65536 --cache-size 1024 --target cache_only --output cache_td_1k.pkl
   python scripts/prebuild_server.py --protocol top_down --num-data 65536 --cache-size 2048 --target cache_only --output cache_td_2k.pkl
   ```
2. **加载阶段**：在 Benchmark 时，同时加载 Base 文件和特定的 Cache 文件。
   ```bash
   # 测试 1K Cache
   python scripts/benchmark_remote.py ... --cache-size 1024 --load-storage "base_td_N65k.pkl,cache_td_1k.pkl"
   
   # 测试 2K Cache (无需重新生成 Base)
   python scripts/benchmark_remote.py ... --cache-size 2048 --load-storage "base_td_N65k.pkl,cache_td_2k.pkl"
   ```
Server 会自动合并（Merge）加载的组件，Client 也会根据文件名或配置识别并跳过初始化，直接使用组合后的状态。
