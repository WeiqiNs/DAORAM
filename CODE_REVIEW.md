# SOMAP 代码审查报告

> 审查日期：2026-02-24  
> 审查分支：`SOMAP` @ `WeiqiNs/DAORAM`  
> 审查范围：`daoram/so/`（核心实现）、`scripts/`（实验脚本）、`tests/`（测试）

---

## 目录

- [仓库结构概览](#仓库结构概览)
- [疑问清单](#疑问清单)
  - [Q1 密文长度不一致（安全性）](#q1-密文长度不一致安全性)
  - [Q2 SORAM dummy access 逻辑（正确性）](#q2-soram-dummy-access-逻辑正确性)
  - [Q3 bottom_adjust.py 重复代码（工程）](#q3-bottom_adjustpy-重复代码工程)
  - [Q4 SOMAPv1 leaf 冲突用 random 替换（安全性）](#q4-somapv1-leaf-冲突用-random-替换安全性)
  - [Q5 SOMAPv1 search vs insert 访问模式不对称（安全性）](#q5-somapv1-search-vs-insert-访问模式不对称安全性)
  - [Q6 残留调试代码（工程）](#q6-残留调试代码工程)
- [实验脚本现状](#实验脚本现状)
- [测试覆盖现状](#测试覆盖现状)

---

## 仓库结构概览

```
daoram/so/                          # SOMAP 核心（4,092 行）
├── soram.py                        # SORAM 基础实现
├── top_down_somap.py               # SOMAPv1（Alg.2）动态 cache
├── top_down_somap_fixed_cache.py   # SOMAPv1 固定 cache + 并行优化
├── bottom_to_up_somap.py           # SOMAPv2（Alg.3）动态 cache
├── bottom_to_up_somap_fixed_cache.py # SOMAPv2 固定 cache + 并行优化 + 动态调整
└── __init__.py

daoram/oram/                        # ORAM 实现
├── static_oram.py                  # Static ORAM（SOMAPv2 的 D_S）
├── path_oram.py                    # Path ORAM 基类
└── ...

daoram/omap/                        # OMAP 实现
├── bplus_ods_omap.py               # B+ 树 OMAP（O_W / O_R 使用）
├── bplus_subset_ods_omap.py        # B+ 子集 OMAP（SOMAPv1 的 O_B）
├── avl_ods_omap.py                 # AVL OMAP
└── ...

scripts/                            # 实验脚本（2,266 行）
├── benchmark_remote.py             # 主 benchmark（支持 BU/TD/Baseline）
├── benchmark_somap_wan.py          # WAN 延迟实验
├── benchmark_somap_small.py        # 小规模本地测试
├── calc_block_size.py              # Block size 计算器
├── prebuild_server.py              # 预构建服务器存储
└── ...

tests/                              # 测试（4,823 行）
├── test_bottom_up_somap.py         # SOMAPv2 测试
├── test_top_down_somap.py          # SOMAPv1 测试
├── test_soram.py                   # SORAM 测试
├── test_orams.py                   # ORAM 测试
└── ...
```

**各实现文件对应关系：**

| 文件 | 论文算法 | 用途 |
|------|---------|------|
| `soram.py` | — | SORAM 基础，PRP + AVL OMAP |
| `top_down_somap.py` | Algorithm 2 | SOMAPv1，动态 cache |
| `top_down_somap_fixed_cache.py` | Algorithm 2 优化 | **实验用主力**，固定 cache，并行化 |
| `bottom_to_up_somap.py` | Algorithm 3 | SOMAPv2，动态 cache |
| `bottom_to_up_somap_fixed_cache.py` | Algorithm 3 优化 | **实验用主力**，固定 cache + 动态调整，并行化 |

---

## 疑问清单

### Q1 密文长度不一致（安全性）

**严重程度：高**

**位置：** 
- `top_down_somap.py:161`
- `bottom_to_up_somap.py:89`

**现状：** 代码中留有未解决的 TODO：

```python
# todo: @weiqi check if all ciphertexts have the same length
# since the value component of dummy pair is "dummy"
```

**具体问题：**

`_encrypt_data()` 直接用 `pickle.dumps()` + AES 加密，没有做定长 padding：

```python
def _encrypt_data(self, data: Any) -> Any:
    serialized_data = pickle.dumps(data)       # 长度取决于 data 内容
    encrypted_data = self._list_cipher.enc(serialized_data)
    return encrypted_data
```

这个函数用于加密 Q_W / Q_R 中的数据。而 Q_W 中存的是：
- 真实条目：`(key, "Key")` 或 `(key, timestamp, "Key")`
- Dummy 条目：`(dummy_index, "Dummy")` 或 `(key, timestamp, "Dummy")`

由于 `"Key"` 和 `"Dummy"` 字符串长度不同（3 vs 5 bytes），且 key 的类型/大小可能不同，pickle 后的长度不一致，加密后密文长度也不同。

**影响：** 服务器可以通过 Q_W / Q_R 条目的密文长度区分 dummy 和真实操作。

**对比：** ORAM 层面的 bucket 加密已经有 `Helper.pad_pickle()` 做了定长 padding，但 `operate_on_list()` 层面没有。

**结论（2026-02-25 审核）：**
- [x] 确认是真实的安全性问题，Q_W/Q_R 存储在服务器端，服务器可通过密文长度区分 dummy 和真实操作
- [x] 已修复：`_encrypt_data()` 中使用 `Helper.pad_pickle()` 统一 padding 到固定长度后再加密
- [x] 新增 `key_length` 和 `value_length` 参数：`key_length` 规定原始 KV 键长度，`value_length` 规定值长度
- [x] padding 长度由 `_compute_list_pad_length()` 自动计算，同时考虑原始 key（`key_length` bytes）和 hash 后的 index（由 `num_data` 决定）
- [x] 修改覆盖 4 个文件，所有测试通过

---

### Q2 SORAM dummy access 逻辑（正确性）

**严重程度：中**

**位置：** `soram.py:265-291`

**现状：** O_W → O_R 迁移时，对 dummy key 的处理：

```python
# 从 Q_W pop 一个 key
key = self.operate_on_list(self._Qw_name, 'pop')

# 如果是 dummy（key >= num_data），从 O_W 删除一个空条目
if key >= self._num_data:
    value = self._Ow.delete(None)
else:
    value = self._Ow.delete(key)

# 将其迁移到 O_R
self.operate_on_list(self._Qr_name, 'insert', data=key)

if key >= self._num_data:
    self._Or.search(key)        # ← 对一个不存在于 O_R 的 dummy key 做 search
else:
    self._Or.insert(key, value)

if key >= self._num_data:
    self.operate_on_list('DB', 'update', pos=self.PRP.encrypt(key), data=value)
    self._dummy_index += 1      # ← 额外增加 dummy_index
    self._dummy_index = self._dummy_index % (2 * self._cache_size)
else:
    self.operate_on_list('DB', 'update', pos=self.PRP.encrypt(key), data=value)
```

**具体问题：**

1. `self._Or.search(key)`：key 是 `>= num_data` 的 dummy index，在 O_R 中不存在。`search` 返回 None。这看起来是为了产生一次 dummy OMAP access 以保证访问模式一致。**但不清楚 search 失败（key 不存在）和 search 成功在 OMAP 层面的访问模式是否一致。**

2. dummy 分支额外增加了 `self._dummy_index`（第 282-283 行），而真实分支没有。这导致 dummy_index 的递增速度与 Case a/b/c 中的不一致。

**结论（2026-02-25 审核）：**
- [x] `search(不存在的key)` 在 c 次以内不构成安全问题：hash(key) 完全随机，cache 保证窗口内不会对同一 key 两次访问 D_S 的相同路径，因此 c 次操作内就是 c 条随机 path，不可区分
- [x] dummy 分支多一次 `_dummy_index += 1` 是有意设计：迁移阶段 dummy item 也要写回 D，需要消耗一个 dummy index 来决定写到 D 的哪个位置
- [x] `soram.py` 是早期版本，实验用 fixed_cache 版本，此处影响较小
- [x] **Q2 已关闭**

---

### Q3 bottom_adjust.py 重复代码（工程）

**严重程度：低**

**位置：** `daoram/so/bottom_adjust.py`（893 行）

**现状：**

`bottom_adjust.py` 定义了 `BottomUpSomapFixedCacheAdjust` 类，与 `bottom_to_up_somap_fixed_cache.py` 中的 `BottomUpSomapFixedCache` 类高度相似。经对比：

- 相同的 `__init__` 参数和结构
- 相同的 `setup()`、`access()`、`_adjust_security_level()` 方法
- 相同的 pending 机制
- 行数差异（893 vs 712），多出的部分可能是 adjust cache size 的额外逻辑

**结论（2026-02-25 审核）：**
- [x] `bottom_adjust.py` 是加速版动态调整实验（push 1 / pop 2 Q_W / pop 3 Q_R），已被新方案取代
- [x] **已删除 `bottom_adjust.py`**
- [x] 新方案：基于时间戳的动态调整（timestamp-based eviction），已实现在 `bottom_to_up_somap_fixed_cache.py` 和 `top_down_somap_fixed_cache.py`
- [x] 核心设计：每个 Q_W/Q_R item 有 push timestamp，驱逐条件 `current_t - push_t >= target_c`，通过 `set_target_cache_size(target_c, adjust_cap)` 动态调整
- [x] cap 机制限制 burst eviction：每步最多 `1 + adjust_cap` 次 pop，防止缩小 c 时一次弹出过多
- [x] 所有 205 个测试通过
- [x] **Q3 已关闭**

---

### Q4 SOMAPv1 leaf 冲突用 random 替换（安全性）

**严重程度：中**

**位置：** `top_down_somap.py:557-563`

**现状：** 在 `_collect_group_leaves_retrieve` 中，当 PRF 生成的 leaf 有重复时：

```python
# Remove duplicates while preserving order
seen = set()
uniq_leaves = []
for l in leaves:
    if l not in seen:
        seen.add(l)
        uniq_leaves.append(l)
    else:
        while True:
            t = random.randint(0, self._num_groups - 1)
            if t not in seen:
                seen.add(t)
                uniq_leaves.append(t)
                break
```

**具体问题：**

1. `random.randint` 是 Python 的伪随机数，**不受 PRF key 控制**。如果 search 和 eviction 中都需要计算同一组 leaves，两次调用可能得到不同的替换结果（因为 random 的状态已变化）。

2. 安全性层面：如果攻击者知道 PRF 有冲突时使用 `random`（而非 PRF 派生），可能影响形式化安全证明。

**对比：** `_collect_group_leaves_generate`（第 566-577 行）**不做去重**，直接返回 PRF 生成的 leaves（允许重复）。两个函数的行为不一致。

**结论（2026-02-25 审核）：**
- [x] `insert` 只取 1 条随机 path，不存在碰撞问题
- [x] `_collect_group_leaves_retrieve` 中的碰撞替换是一次性的（同一次调用内读+写），不需要跨操作重放，不影响确定性
- [x] **已修复**：`random.randint` 已替换为 `secrets.randbelow`（密码学安全随机数），覆盖 `top_down_somap.py` 和 `top_down_somap_fixed_cache.py`
- [x] **Q4 已关闭**

---

### Q5 SOMAPv1 search vs insert 访问模式不对称（安全性）

**严重程度：中**

**位置：** `top_down_somap.py:579-653`

**现状：**

- `search()` 方法读写 `upper_bound` 条路径（一整个 group 的所有 leaf）
- `insert()` 方法只读写 **1 条随机路径**

```python
def search(self, key, seed):
    retrieve_leaves = self._collect_group_leaves_retrieve(group_index, seed)  # upper_bound 条
    raw_paths = self._client.read_query(label=self._Tree_name, leaf=retrieve_leaves)
    ...

def insert(self, key, value, seed):
    leaves = [random.randint(0, self._num_groups - 1)]  # 只有 1 条
    raw_paths = self._client.read_query(label=self._Tree_name, leaf=leaves)
    ...
```

**具体问题：**

服务器可以通过观察每次操作读取的路径数量（upper_bound vs 1）来区分 search 和 insert。这直接破坏了 snapshot obliviousness 的要求——窗口内的操作应该是不可区分的。

**结论（2026-02-25 审核）：**
- [x] 这是设计上的已知属性：search 和 insert 确实允许被区分
- [ ] 未来有时间时统一 path 数量（将 insert 也 padding 到 upper_bound），代码中已加 TODO 标记
- [x] fixed_cache 版本同样存在此问题

---

### Q6 残留调试代码（工程）

**严重程度：低**

**结论（2026-02-25 审核）：**
- [x] 移除所有 `print()` 调试输出：`soram.py` 的 Qw/Qr 泄漏、`top_down_somap_fixed_cache.py` 的初始化/DEBUG print、`bottom_to_up_somap_fixed_cache.py` 的 DEBUG print
- [x] 修正 `soram.py` 拼写错误 `"unkonw"` → `"unknown"`
- [x] 所有 `print("error: ...")` 改为 `raise ValueError(...)`
- [x] `[WARNING]` print 改为 `warnings.warn()`
- [x] OMAP `restore_client_state` 的 print 改为 `logging.debug()`
- [x] 清理 `flexible_binary_tree.py`、`interact_server.py` 中的调试 print
- [x] 移除所有注释掉的 `# print(...)` 残留
- [x] 仅保留 `crypto.py` 的 `__main__` 测试块 print（不影响正常运行）
- [x] 82 个测试全部通过
- [x] **Q6 已关闭**

---

## 实验脚本现状

### 主力 benchmark：`scripts/benchmark_remote.py`

**已支持：**
- 三种协议：Bottom-Up（SOMAPv2）、Top-Down（SOMAPv1）、BPlus OMAP Baseline
- 参数：`--num-data`、`--cache-size`、`--value-size`、`--order`、`--num-ops`
- 延迟模拟：`--latency-ms`（支持 30/50/80ms）
- 指标收集：rounds、bandwidth（sent/recv）、elapsed time、client/server storage
- 预构建存储：`--load-storage`（跳过 setup 加速实验）

**用于论文的实验参数范围（需确认）：**
- `num_data`：2^12 ~ 2^24
- `cache_size`：论文中的 c 值
- `value_size`：论文中固定一个值（16？256？）
- `latency_ms`：30 / 50 / 80

### 缺少的内容

1. **没有自动批量跑不同 block size 的脚本**  
   `--value-size` 参数已存在，但没有 wrapper 脚本自动遍历 {64B, 256B, 1KB, 4KB, 16KB}

2. **没有 Zipf 分布访问模式的实验脚本**  
   benchmark_remote.py 的 ops 生成是均匀随机的，论文中的安全性评估需要 Zipf(θ=0.99) 分布

3. **没有存储开销的独立统计脚本**  
   server_storage 的计算目前是理论估算（`calc_server_storage_*`），不是实际测量

---

## 测试覆盖现状

### 已覆盖

| 测试文件 | 覆盖内容 | 测试规模 |
|---------|---------|---------|
| `test_bottom_up_somap.py` | CRUD、cache overflow、动态调 c、加密、边界、性能 | n ≤ 512, c ≤ 20 |
| `test_top_down_somap.py` | search/insert、cache 管理、加密 | n ≤ 512 |
| `test_soram.py` | 基本 read/write | n ≤ 100 |
| `test_orams.py` | Path ORAM / DA-ORAM 等 | — |
| `test_omaps.py` | AVL / B+ 各种 OMAP | — |

### 未覆盖

1. **安全性验证**：没有测试"相同操作序列在不同数据下产生相同访问模式"
2. **大规模正确性**：最大测试 n=512，论文实验到 n=2^24
3. **Static ORAM 单独测试**：无 `test_static_oram.py`
4. **fixed_cache 版本的测试**：没有看到 `test_bottom_up_somap_fixed_cache.py`，而 fixed_cache 才是实验主力

---

## 已完成的改进

| 项目 | 描述 | 涉及文件 |
|------|------|---------|
| **Q1 修复** | `_encrypt_data()` 加 `pad_pickle()` 定长 padding | 4 个 SOMAP 文件 |
| **Q2 关闭** | `search(不存在key)` 在 c 窗口内不可区分 | — |
| **Q3 完成** | 删除 `bottom_adjust.py`；两个 fixed_cache 加 timestamp-based 动态调整 | `bottom_to_up_somap_fixed_cache.py`, `top_down_somap_fixed_cache.py` |
| **Q4 修复** | `random.randint` → `secrets.randbelow` | `top_down_somap.py`, `top_down_somap_fixed_cache.py` |
| **动态轮次** | OMAP 交互轮次按 `max(\|Q_W\|, \|Q_R\|)` 动态决定，不再 pad 到 `_max_height` | `tree_ods_omap.py`, `bplus_ods_omap.py`, `bplus_subset_ods_omap.py`, 两个 fixed_cache |

### 动态轮次设计说明

O_W/O_R（及 O_B）的 OMAP 树用 `num_data`（最大数据库大小）初始化存储容量，但每次查询的交互轮次 pad 到 `effective_height = ceil(log(max(|Q_W|, |Q_R|, 1), ceil(order/2))) + 1`。

安全性论证：泄露轮次变化不引入额外信息，因为：
1. `c` 是公开参数
2. Q_W、Q_R 存储在服务端，其长度对云服务器本身就是可见的

---

## 待完成事项

### 代码层

| 优先级 | 项目 | 描述 |
|--------|------|------|
| 中 | **Q5 路径数对称** | `insert` 也 pad 到 `upper_bound` 条路径，消除 search/insert 可区分性 |
| 低 | **Q6 调试代码清理** | 去除 soram.py、top_down/bottom_up 中的残留 print，或统一换成 logging |
| 中 | **TODO: 动态 ORAM 树大小** | 目前 O_W/O_R/O_B 的 ORAM 底层树按 `num_data` 初始化，当 `cache_size << num_data` 时浪费严重。应支持按当前实际 cache 大小动态 resize ORAM 树（类似 dynamic array doubling/halving），在 `set_target_cache_size` 时触发 |

### 实验层

| 项目 | 描述 |
|------|------|
| **batch block size 脚本** | 自动遍历 {64B, 256B, 1KB, 4KB, 16KB} 跑 benchmark |
| **Zipf 访问模式** | 在 benchmark_remote.py 中支持 Zipf(θ=0.99) 分布的操作序列 |
| **实际存储开销测量** | 用实际运行数据替代理论估算的 server_storage |

### 测试层

| 项目 | 描述 |
|------|------|
| **fixed_cache 正确性测试** | 补充 `test_bottom_up_somap_fixed_cache.py` 和 `test_top_down_somap_fixed_cache.py` |
| **安全性验证** | 测试相同操作序列在不同数据下的访问模式一致性 |
| **大规模测试** | 目前最大 n=512，论文实验到 n=2^24 |
