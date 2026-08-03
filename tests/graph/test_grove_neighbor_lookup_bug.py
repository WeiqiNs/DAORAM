"""
专门测试 Grove 中 neighbor query 和 lookup 交互的 bug。

发现的问题：
1. neighbor 函数重复创建 Type 2 dup（带有过时的 graph_leaf）
2. _pos_meta 没有被正确写回，导致 AVL node 的 graph_leaf 没有更新

复现场景：
- 有四个连接的点 {0,1,2,3} 存储于 grove 中
- 第一次运行 neighbor(0) 时，会发现有一个 type 2 的 dup 被创建给 vertex 0
- 这个 type 2 的 dup 的 graph_path 是旧路径 B 而不是新路径 A
- 说明 type 2 的 dup 是在 neighbor query 中的 lookup(0) 时创建的，但 neighbor 又重复创建了一遍
"""

import pytest
from daoram.graph.grove import Grove
from daoram.dependency import InteractLocalServer, Data


class TestNeighborLookupBug:
    """测试 neighbor 查询和 lookup 之间的 bug。"""

    def create_grove(self):
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
        grove.init_server_storage()
        return grove

    def setup_four_connected_vertices(self, grove):
        """
        创建四个连接的顶点 {0, 1, 2, 3}。
        
        图结构:
            0 --- 1
            |     |
            2 --- 3
        """
        grove.insert(vertex=(0, "vertex_0", {}))
        grove.insert(vertex=(1, "vertex_1", {0: None}))
        grove.insert(vertex=(2, "vertex_2", {0: None}))
        grove.insert(vertex=(3, "vertex_3", {1: None, 2: None}))
        return grove

    def test_neighbor_then_lookup_same_vertex(self):
        """
        测试 neighbor(0) 之后 lookup(0) 的正确性。
        
        这是 bug 的核心场景：
        1. neighbor(0) 内部调用 lookup(0) 获取邻居列表
        2. neighbor 又为 vertex 0 创建了重复的 type 2 dup（使用旧的 graph_leaf）
        3. 之后的 lookup(0) 可能无法正确找到 vertex 0
        """
        grove = self.create_grove()
        self.setup_four_connected_vertices(grove)
        
        # 记录操作前 vertex 0 在 Graph ORAM 中的位置
        # 通过 lookup 获取
        result_before = grove.lookup(keys=[0])
        assert 0 in result_before, "lookup(0) 应该在 neighbor 之前成功"
        
        # 执行 neighbor(0) 查询
        neighbors = grove.neighbor(keys=[0])
        assert len(neighbors) > 0, "vertex 0 应该有邻居"
        
        # 关键测试：neighbor 之后的 lookup(0)
        result_after = grove.lookup(keys=[0])
        assert 0 in result_after, (
            "BUG: lookup(0) 在 neighbor(0) 之后失败！"
            "这可能是因为 neighbor 重复创建了带有过时 graph_leaf 的 type 2 dup"
        )
        
        # 验证数据完整性
        assert result_after[0][0] == "vertex_0", "vertex 0 的数据应该保持不变"

    def test_repeated_neighbor_lookup_cycle(self):
        """
        测试多次 neighbor 和 lookup 的循环。
        
        每次循环都可能累积错误的 type 2 dups。
        """
        grove = self.create_grove()
        self.setup_four_connected_vertices(grove)
        
        for i in range(5):
            # neighbor 查询
            neighbors = grove.neighbor(keys=[0])
            
            # lookup 应该继续工作
            result = grove.lookup(keys=[0])
            assert 0 in result, f"循环 {i}: lookup(0) 在 neighbor(0) 之后失败"
            
            # 验证数据完整性
            assert result[0][0] == "vertex_0", f"循环 {i}: vertex 0 数据损坏"

    def test_neighbor_then_lookup_neighbor(self):
        """
        测试 neighbor(0) 之后 lookup 邻居顶点。
        
        邻居的 adjacency 列表应该包含更新后的 vertex 0 的 graph_leaf。
        """
        grove = self.create_grove()
        self.setup_four_connected_vertices(grove)
        
        # neighbor(0) 应该返回 vertex 1, 2
        neighbors = grove.neighbor(keys=[0])
        neighbor_keys = list(neighbors.keys())
        
        # lookup 每个邻居
        for nk in neighbor_keys:
            result = grove.lookup(keys=[nk])
            assert nk in result, f"lookup({nk}) 在 neighbor(0) 之后失败"
            
            # 邻居的 adjacency 列表应该包含 vertex 0
            adjacency = result[nk][1]
            assert 0 in adjacency, f"vertex {nk} 的邻居列表应该包含 vertex 0"


class TestType2DupDiagnosis:
    """诊断 Type 2 dup 重复创建的问题。"""

    def create_grove(self):
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
        grove.init_server_storage()
        return grove

    def test_graph_meta_dup_after_neighbor(self):
        """
        检查 neighbor 之后 _graph_meta.stash 中是否有重复的 type 2 dup。
        
        问题：neighbor 函数在 lookup 之后又为 center_key 创建了 type 2 dup，
        但使用的是旧的 graph_leaf（来自 visited_nodes_map）。
        """
        grove = self.create_grove()
        
        # 创建简单的图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None}))
        
        # 执行 neighbor(0)
        grove.neighbor(keys=[0])
        
        # 检查 _graph_meta.stash 中是否有针对 vertex 0 的 type 2 dup
        type2_dups_for_vertex_0 = []
        for dup in grove._graph_meta.stash:
            if dup.key == 0 and not isinstance(dup.value, tuple):
                type2_dups_for_vertex_0.append(dup)
        
        # 理想情况下，neighbor 之后不应该有针对 center_key 的 type 2 dup 残留
        # 因为 lookup 已经处理过了
        print(f"DEBUG: neighbor 后 vertex 0 的 type 2 dups 数量: {len(type2_dups_for_vertex_0)}")
        for dup in type2_dups_for_vertex_0:
            print(f"  - dup.key={dup.key}, dup.leaf={dup.leaf}, dup.value={dup.value}")
        
        # 这个测试暴露问题但不 assert 失败，因为我们是在诊断

    def test_pos_meta_dup_creation(self):
        """
        检查 _pos_meta 中的 dup 是否被正确创建和应用。
        
        问题：lookup_without_omap 和 neighbor 创建的 pos_meta_duplications
        没有被立即写回，导致 AVL node 的 graph_leaf 没有更新。
        """
        grove = self.create_grove()
        
        # 创建简单的图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        # 记录 lookup 前 _pos_meta.stash 的状态
        pos_meta_before = len(grove._pos_meta.stash)
        
        # 执行 lookup(0)
        result = grove.lookup(keys=[0])
        
        # 检查 _pos_meta.stash 是否有新的 dup
        pos_meta_after = len(grove._pos_meta.stash)
        print(f"DEBUG: lookup 后 _pos_meta.stash 变化: {pos_meta_before} -> {pos_meta_after}")
        
        # 打印 stash 中的 dup 详情
        for dup in grove._pos_meta.stash:
            print(f"  - dup.key={dup.key}, dup.leaf={dup.leaf}, dup.value={dup.value}")


class TestAVLNodeGraphLeafConsistency:
    """测试 AVL node 中存储的 graph_leaf 与实际 vertex 位置的一致性。"""

    def create_grove(self):
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
        grove.init_server_storage()
        return grove

    def test_graph_leaf_consistency_after_operations(self):
        """
        验证 AVL node 存储的 graph_leaf 是否与 vertex 实际位置一致。
        
        问题：pos_meta dup 没有被正确写回和应用，导致 AVL node 的 graph_leaf
        是旧值，而 vertex 实际上已经移动到新位置了。
        """
        grove = self.create_grove()
        
        # 创建图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None}))
        
        # 执行 neighbor(0)
        grove.neighbor(keys=[0])
        
        # 通过 batch_search 获取 AVL node 中存储的 graph_leaf
        avl_result, visited_nodes, _ = grove._pos_omap.batch_search(
            keys=[0], return_visited_nodes=True
        )
        
        avl_stored_graph_leaf = avl_result.get(0)
        print(f"DEBUG: AVL node 存储的 vertex 0 graph_leaf: {avl_stored_graph_leaf}")
        
        # 查找 vertex 0 在 _graph_oram.stash 中的实际 leaf
        actual_leaf = None
        for data in grove._graph_oram.stash:
            if data.key == 0:
                actual_leaf = data.leaf
                break
        print(f"DEBUG: vertex 0 在 stash 中的实际 leaf: {actual_leaf}")
        
        # 如果 vertex 不在 stash 中，需要从 storage 读取
        if actual_leaf is None:
            print("DEBUG: vertex 0 不在 _graph_oram.stash 中，需要从 storage 读取")


class TestMixedOperationsStress:
    """混合操作的压力测试，特别关注 neighbor 和 lookup 的交互。"""

    def create_grove(self):
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
        grove.init_server_storage()
        return grove

    def test_insert_neighbor_lookup_pattern(self):
        """
        测试 insert -> neighbor -> lookup 模式。
        
        这个模式特别容易触发 bug，因为：
        1. insert 会创建新的 vertex 和 adjacency 关系
        2. neighbor 会读取并更新 adjacency
        3. lookup 需要找到正确位置的 vertex
        """
        grove = self.create_grove()
        
        # 逐步插入顶点
        for i in range(4):
            neighbors = {j: None for j in range(i)}
            grove.insert(vertex=(i, f"v{i}", neighbors))
            print(f"插入 vertex {i}")
            
            # 每次插入后，对已存在的顶点执行 neighbor + lookup
            for j in range(i + 1):
                if i > 0:  # 只有当有邻居时才做 neighbor
                    grove.neighbor(keys=[j])
                
                result = grove.lookup(keys=[j])
                assert j in result, f"lookup({j}) 在插入 vertex {i} 后失败"

    @pytest.mark.parametrize("pattern", [
        ["neighbor", "lookup", "lookup"],
        ["lookup", "neighbor", "lookup"],
        ["neighbor", "neighbor", "lookup"],
        ["lookup", "lookup", "neighbor", "lookup"],
    ])
    def test_operation_patterns(self, pattern):
        """
        测试不同的操作模式。
        
        不同的操作序列可能导致不同的 dup 累积和处理问题。
        """
        grove = self.create_grove()
        
        # 创建图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None}))
        
        for op in pattern:
            if op == "neighbor":
                grove.neighbor(keys=[0])
            elif op == "lookup":
                result = grove.lookup(keys=[0])
                assert 0 in result, f"lookup(0) 在模式 {pattern} 中失败"

    def test_all_vertices_accessible_after_neighbor_queries(self):
        """
        测试对所有顶点执行 neighbor 查询后，所有顶点仍然可访问。
        """
        grove = self.create_grove()
        
        # 创建一个较大的图
        n = 8
        for i in range(n):
            neighbors = {j: None for j in range(max(0, i-2), i)}
            grove.insert(vertex=(i, f"v{i}", neighbors))
        
        # 对每个顶点执行 neighbor 查询
        for i in range(n):
            grove.neighbor(keys=[i])
        
        # 验证所有顶点仍然可访问
        for i in range(n):
            result = grove.lookup(keys=[i])
            assert i in result, f"vertex {i} 在 neighbor 查询后不可访问"


class TestPosMetaWriteBack:
    """测试 _pos_meta 的写回逻辑。"""

    def create_grove(self):
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
        grove.init_server_storage()
        return grove

    def test_pos_meta_stash_growth(self):
        """
        监控 _pos_meta.stash 的增长。
        
        如果 _pos_meta 没有被正确写回，stash 会不断增长。
        """
        grove = self.create_grove()
        
        # 创建图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        stash_sizes = []
        
        for i in range(10):
            # 执行操作
            grove.lookup(keys=[0])
            stash_sizes.append(len(grove._pos_meta.stash))
        
        print(f"DEBUG: _pos_meta.stash 大小变化: {stash_sizes}")
        
        # 如果 stash 在不断增长且没有减少，说明写回有问题
        # 注意：正常情况下 stash 大小应该在合理范围内波动

    def test_pos_meta_eviction_paths(self):
        """
        检查 _pos_meta 的 eviction 是否使用正确的路径。
        
        问题：在 lookup_without_omap 中，只调用了 _graph_oram.queue_write()
        和 _graph_meta.queue_write()，没有调用 _pos_meta.queue_write()。
        """
        grove = self.create_grove()
        
        # 创建图
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        
        # 执行 lookup
        grove.lookup(keys=[0])
        
        # 检查 _pos_meta.stash 中的 dup
        print(f"DEBUG: lookup 后 _pos_meta.stash 内容:")
        for dup in grove._pos_meta.stash:
            print(f"  - key={dup.key}, leaf={dup.leaf}, value={dup.value}")
        
        # 再次执行 lookup（触发 batch_search，应该读写 _pos_meta）
        grove.lookup(keys=[0])
        
        print(f"DEBUG: 第二次 lookup 后 _pos_meta.stash 内容:")
        for dup in grove._pos_meta.stash:
            print(f"  - key={dup.key}, leaf={dup.leaf}, value={dup.value}")


if __name__ == "__main__":
    # 可以直接运行进行调试
    import sys
    
    print("=" * 60)
    print("运行诊断测试")
    print("=" * 60)
    
    test = TestType2DupDiagnosis()
    print("\n--- test_graph_meta_dup_after_neighbor ---")
    test.test_graph_meta_dup_after_neighbor()
    
    print("\n--- test_pos_meta_dup_creation ---")
    test.test_pos_meta_dup_creation()
    
    print("\n" + "=" * 60)
    print("运行一致性测试")
    print("=" * 60)
    
    test2 = TestAVLNodeGraphLeafConsistency()
    print("\n--- test_graph_leaf_consistency_after_operations ---")
    test2.test_graph_leaf_consistency_after_operations()
    
    print("\n" + "=" * 60)
    print("运行 PosMetaWriteBack 测试")
    print("=" * 60)
    
    test3 = TestPosMetaWriteBack()
    print("\n--- test_pos_meta_stash_growth ---")
    test3.test_pos_meta_stash_growth()
    
    print("\n--- test_pos_meta_eviction_paths ---")
    test3.test_pos_meta_eviction_paths()
    
    print("\n" + "=" * 60)
    print("运行核心 bug 测试")
    print("=" * 60)
    
    test4 = TestNeighborLookupBug()
    print("\n--- test_neighbor_then_lookup_same_vertex ---")
    try:
        test4.test_neighbor_then_lookup_same_vertex()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
