"""
精确测试 _pos_meta eviction 问题。

问题根因：
1. lookup_without_omap 在 batch_search 之后创建 pos_meta dups
2. 这些 dups 被加入 _pos_meta.stash
3. 但 lookup_without_omap 只写回 _graph_oram 和 _graph_meta
4. 下次 batch_search 时，这些 dups 可能被 evict 到错误的位置
5. 导致 AVL node 的 graph_leaf 永远不会被更新

测试场景：复现 debug_stress_3 的 Step 21 失败
"""

import random
import pytest
from daoram.graph.grove import Grove
from daoram.dependency import InteractLocalServer, Data


def create_grove():
    """创建一个用于测试的 Grove 实例。"""
    client = InteractLocalServer()
    grove = Grove(
        max_deg=5,
        num_opr=100,
        num_data=1024,
        key_size=16,
        data_size=64,
        client=client,
        encryptor=None,
        stash_scale=20,
    )
    grove._pos_omap.init_server_storage()
    grove._graph_oram.init_server_storage()
    grove._graph_meta.init_server_storage()
    grove._pos_meta.init_server_storage()
    return grove


class TestPosMetaEvictionBug:
    """测试 _pos_meta eviction 问题。"""

    def test_exact_failure_scenario_seed_3(self):
        """
        精确复现 debug_stress_3 的失败场景（trial=3, seed=300）。
        
        失败发生在 Step 21: neighbor(3) 之后 lookup(3) 失败。
        """
        random.seed(300)  # trial * 100 = 3 * 100 = 300
        grove = create_grove()

        inserted = set()
        deleted = set()

        for step in range(22):  # 运行到 Step 21
            op = random.randint(0, 6)
            active = list(inserted - deleted)

            if op == 0 or len(active) == 0:
                new_key = len(inserted)
                neighbors = {}
                if active:
                    n = min(len(active), random.randint(1, 3))
                    random.shuffle(active)
                    neighbors = {k: None for k in active[:n]}
                grove.insert((new_key, f"d{new_key}", neighbors))
                inserted.add(new_key)

            elif op == 1 and len(active) > 2:
                key = random.choice(active)
                grove.delete(key)
                deleted.add(key)

            elif op == 2 and active:
                key = random.choice(active)
                result = grove.lookup([key])
                assert key in result, f"Step {step}: lookup({key}) failed!"
                
            elif op == 3 and active:
                key = random.choice(active)
                grove.neighbor([key])
                result = grove.lookup([key])
                assert key in result, f"Step {step}: lookup({key}) after neighbor failed!"

            elif op == 4 and active:
                key = random.choice(active)
                result = grove.t_hop(key, random.randint(1, 2))
                assert key in result, f"Step {step}: t_hop missing start {key}"

            elif op == 5 and active:
                key = random.choice(active)
                result = grove.t_traversal(key, random.randint(1, 2))
                assert key in result, f"Step {step}: t_traversal missing start {key}"

            else:
                if active:
                    key = random.choice(active)
                    grove.lookup([key])

    def test_simplified_failure_scenario(self):
        """
        简化的失败场景，不使用随机操作。
        
        1. 创建一些顶点
        2. 执行 t_traversal（内部多次调用 neighbor）
        3. 验证所有顶点仍然可访问
        """
        grove = create_grove()
        
        # 创建顶点 0, 1 相连
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        # 执行 t_traversal，内部会调用多次 neighbor
        for _ in range(5):
            result = grove.t_traversal(key=0, num_hop=2)
            assert 0 in result, "t_traversal should include start vertex"
        
        # 验证所有顶点可访问
        for key in [0, 1]:
            result = grove.lookup(keys=[key])
            assert key in result, f"vertex {key} 不可访问"

    def test_pos_meta_dup_persistence(self):
        """
        测试 pos_meta dups 是否在多次操作后仍然有效。
        """
        grove = create_grove()
        
        # 创建顶点
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        # 执行一系列操作
        for i in range(10):
            # 交替执行 lookup 和 neighbor
            if i % 2 == 0:
                grove.lookup(keys=[0])
            else:
                grove.neighbor(keys=[0])
            
            # 每次操作后验证 vertex 1 可访问
            result = grove.lookup(keys=[1])
            assert 1 in result, f"iteration {i}: vertex 1 不可访问"

    def test_neighbor_lookup_interleave(self):
        """
        测试 neighbor 和 lookup 交替执行。
        """
        grove = create_grove()
        
        # 创建多个相连的顶点
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None, 1: None}))
        grove.insert(vertex=(3, "v3", {1: None}))
        
        # 交替执行 neighbor 和 lookup
        for i in range(20):
            key = i % 4
            
            if i % 3 == 0:
                grove.neighbor(keys=[key])
            
            result = grove.lookup(keys=[key])
            assert key in result, f"iteration {i}: vertex {key} 不可访问"


class TestPosMetaWriteBack:
    """测试 _pos_meta 写回逻辑。"""

    def test_pos_meta_written_after_lookup(self):
        """
        验证 lookup_without_omap 之后 _pos_meta 应该被写回。
        
        当前代码的问题：只写回 _graph_oram 和 _graph_meta，没有写回 _pos_meta。
        """
        grove = create_grove()
        
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        # 记录 lookup 前的 _pos_meta.stash 大小
        before_size = len(grove._pos_meta.stash)
        
        # 执行 lookup
        grove.lookup(keys=[0])
        
        # lookup 应该创建 pos_meta dup
        after_size = len(grove._pos_meta.stash)
        
        # 问题：这些 dups 留在 stash 中，没有被写回
        # 理想情况：lookup_without_omap 应该也写回 _pos_meta
        print(f"_pos_meta.stash: {before_size} -> {after_size}")

    def test_pos_meta_dup_leaf_matches_read_path(self):
        """
        验证 pos_meta dup 的 leaf 是否与下次读取的路径匹配。
        """
        grove = create_grove()
        
        grove.insert(vertex=(0, "v0", {}))
        
        # 获取 AVL node 0 的当前 pos_leaf
        avl_result, visited, _ = grove._pos_omap.batch_search(
            keys=[0], return_visited_nodes=True
        )
        initial_pos_leaf = visited[0][0] if 0 in visited else None
        print(f"Initial pos_leaf for vertex 0: {initial_pos_leaf}")
        
        # 执行 lookup
        grove.lookup(keys=[0])
        
        # 检查 _pos_meta.stash 中是否有 dup
        for dup in grove._pos_meta.stash:
            if dup.key == 0:
                print(f"pos_meta dup: key={dup.key}, leaf={dup.leaf}, value={dup.value}")
                
        # 再次获取 AVL node 0 的 pos_leaf
        avl_result2, visited2, _ = grove._pos_omap.batch_search(
            keys=[0], return_visited_nodes=True
        )
        new_pos_leaf = visited2[0][0] if 0 in visited2 else None
        print(f"New pos_leaf for vertex 0: {new_pos_leaf}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试精确失败场景")
    print("=" * 60)
    
    test = TestPosMetaEvictionBug()
    
    print("\n--- test_exact_failure_scenario_seed_3 ---")
    try:
        test.test_exact_failure_scenario_seed_3()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
    
    print("\n--- test_simplified_failure_scenario ---")
    try:
        test.test_simplified_failure_scenario()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
    
    print("\n--- test_pos_meta_dup_persistence ---")
    try:
        test.test_pos_meta_dup_persistence()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
    
    print("\n--- test_neighbor_lookup_interleave ---")
    try:
        test.test_neighbor_lookup_interleave()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
