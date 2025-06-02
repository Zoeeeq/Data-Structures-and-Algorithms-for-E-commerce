class Customer:
    """表示网络中的一个客户"""
    
    def __init__(self, name):
        """
        初始化一个客户
        
        参数:
            name (str): 客户名称
        """
        self.name = name
    
    def __str__(self):
        """客户的字符串表示"""
        return f"客户: {self.name}"


class CustomerNetwork:
    """
    表示客户关系网络的加权有向图
    """
    
    def __init__(self):
        """初始化空的客户网络"""
        self.customers = {}  # 客户名称到Customer对象的映射
        self.graph = {}  # 邻接表
    
    def add_customer(self, customer):
        """
        向网络中添加新客户
        
        参数:
            customer: 要添加的客户
            
        返回:
            bool: 添加成功返回True，客户已存在返回False
        """
        if customer.name in self.customers:
            return False
        
        self.customers[customer.name] = customer
        self.graph[customer.name] = []
        return True
    
    def add_relationship(self, from_customer, to_customer, weight):
        """
        添加或更新从一个客户到另一个客户的关系
        
        参数:
            from_customer: 施加影响的客户名称
            to_customer: 被影响的客户名称
            weight: 影响力强度 (0-1)
            
        返回:
            bool: 成功返回True，如果客户不存在则返回False
        """
        if from_customer not in self.graph or to_customer not in self.graph:
            return False
        
        # 检查关系是否已存在并更新
        for i, (neighbor, _) in enumerate(self.graph[from_customer]):
            if neighbor == to_customer:
                self.graph[from_customer][i] = (to_customer, weight)
                return True
        
        # 添加新关系
        self.graph[from_customer].append((to_customer, weight))
        return True
    
    def calculate_customer_importance(self):
        """
        使用PageRank算法计算每个客户的重要性
        
        返回:
            dict: 客户名称到重要性得分的映射
        """
        num_customers = len(self.customers)
        if num_customers == 0:
            return {}
        
        # 初始化PageRank值
        page_rank = {customer: 1.0 / num_customers for customer in self.graph}
        damping = 0.85  # 阻尼系数
        iterations = 30  # 迭代次数
        
        for _ in range(iterations):
            new_rank = {customer: (1 - damping) / num_customers for customer in self.graph}
            
            # 基于入链更新PageRank
            for customer, neighbors in self.graph.items():
                outgoing_links = len(neighbors)
                if outgoing_links > 0:
                    outgoing_sum = sum(weight for _, weight in neighbors)
                    for neighbor, weight in neighbors:
                        # 在PageRank计算中考虑权重
                        new_rank[neighbor] += damping * page_rank[customer] * weight / outgoing_sum
            
            page_rank = new_rank
        
        return page_rank
    
    def find_influenced_customers(self, customer_name):
        """
        找出一个客户可以影响的所有客户
        
        参数:
            customer_name: 施加影响的客户名称
            
        返回:
            set: 受影响客户名称的集合
        """
        if customer_name not in self.graph:
            return set()
        
        influenced = set()
        queue = [customer_name]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            
            if current != customer_name:
                influenced.add(current)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor, _ in self.graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return influenced
    
    def find_influenced_customers_with_limits(self, customer_name, max_depth=float('inf'), min_influence=0):
        """
        限制路径长度和最小影响力门槛，找出受影响的客户
        
        参数:
            customer_name: 施加影响的客户名称
            max_depth: 考虑的最大路径长度
            min_influence: 最小影响力阈值
            
        返回:
            dict: 客户名称到影响力水平的映射
        """
        if customer_name not in self.graph:
            return {}
        
        influenced = {}
        queue = [(customer_name, 1.0, 0)]  # (客户, 影响力, 深度)
        visited = set()
        
        while queue:
            current, influence, depth = queue.pop(0)
            
            if current != customer_name:
                influenced[current] = influence
            
            if depth >= max_depth:
                continue
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor, weight in self.graph[current]:
                new_influence = influence * weight  # 影响力的乘法衰减
                if new_influence >= min_influence and neighbor not in visited:
                    queue.append((neighbor, new_influence, depth + 1))
        
        return influenced
    
    def __str__(self):
        """客户网络的字符串表示"""
        result = [f"客户网络共有 {len(self.customers)} 个客户:"]
        
        for customer, neighbors in self.graph.items():
            if neighbors:
                neighbor_str = ", ".join(f"{neighbor}({weight:.2f})" for neighbor, weight in neighbors)
                result.append(f"{customer} 影响: {neighbor_str}")
            else:
                result.append(f"{customer} (无向外影响)")
        
        return "\n".join(result)
    
    def calculate_transitive_closure(self):
        """
        使用Floyd-Warshall算法计算客户网络的传递闭包
        
        返回:
            dict: 客户名称到可达客户集合的映射
        """
        # 初始化传递闭包矩阵
        closure = {}
        for customer in self.customers:
            closure[customer] = set()
            # 添加直接连接的客户
            for neighbor, _ in self.graph.get(customer, []):
                closure[customer].add(neighbor)
        
        # Floyd-Warshall算法 - 对每个中间节点k
        for k in self.customers:
            # 对每个起点i
            for i in self.customers:
                # 对每个终点j
                for j in self.customers:
                    # 如果i可以到达k且k可以到达j，那么i可以到达j
                    if k in closure[i] and j in closure[k]:
                        closure[i].add(j)
        
        return closure

    def find_minimum_spanning_tree(self, algorithm='prim'):
        """
        从客户网络中提取最小生成树，发现关键影响路径
        
        参数:
            algorithm: 使用的算法，'prim'或'kruskal'
            
        返回:
            list: 最小生成树中的边列表，每个边是(from_customer, to_customer, weight)
        """
        if not self.customers:
            return []
        
        # 将有向图转换为无向图（取平均影响力）
        undirected_graph = {}
        for u in self.customers:
            undirected_graph[u] = {}
        
        # 添加边并计算平均权重
        for u in self.customers:
            for v, weight_uv in self.graph.get(u, []):
                # 查找反向边的权重
                weight_vu = 0
                for v_neighbor, v_weight in self.graph.get(v, []):
                    if v_neighbor == u:
                        weight_vu = v_weight
                        break
                
                # 计算平均权重
                avg_weight = (weight_uv + weight_vu) / 2
                
                # 添加到无向图
                undirected_graph[u][v] = avg_weight
                undirected_graph[v][u] = avg_weight
        
        # 根据选择的算法执行相应的MST计算
        if algorithm.lower() == 'prim':
            return self._prim_mst(undirected_graph)
        elif algorithm.lower() == 'kruskal':
            return self._kruskal_mst(undirected_graph)
        

    def _prim_mst(self, graph):
        """
        使用Prim算法计算最小生成树
        
        参数:
            graph (dict): 无向图的邻接表表示
            
        返回:
            list: 最小生成树中的边列表，每个边是(from_customer, to_customer, weight)
        """
        if not graph:
            return []
            
        # 任选一个起始顶点
        start_vertex = next(iter(graph))
        
        # 初始化MST边集和已访问顶点集
        mst_edges = []
        visited = {start_vertex}
        
        # 使用列表作为优先队列跟踪所有可能的边
        edges = []
        for neighbor, weight in graph[start_vertex].items():
            edges.append((weight, start_vertex, neighbor))
        
        # 按权重排序
        edges.sort()
        
        # 当还有顶点未加入MST且还有边可以考虑时，继续添加边
        while edges and len(visited) < len(graph):
            weight, u, v = edges.pop(0)  # 获取权重最小的边
            
            if v not in visited:  # 如果v未加入MST
                visited.add(v)
                mst_edges.append((u, v, weight))
                
                # 添加v的所有边到候选边集
                for neighbor, weight in graph[v].items():
                    if neighbor not in visited:
                        edges.append((weight, v, neighbor))
                
                edges.sort()  # 保持边按权重排序
        
        return mst_edges

    def _kruskal_mst(self, graph):
        """
        使用Kruskal算法计算最小生成树
        
        参数:
            graph (dict): 无向图的邻接表表示
            
        返回:
            list: 最小生成树中的边列表，每个边是(from_customer, to_customer, weight)
        """
        # 收集所有边
        edges = []
        for u in graph:
            for v, weight in graph[u].items():
                if u < v:  # 避免重复添加同一条边
                    edges.append((weight, u, v))
        
        # 按权重排序
        edges.sort()
        
        # 初始化并查集
        parent = {customer: customer for customer in graph}
        
        # 查找函数
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # 路径压缩
            return parent[x]
        
        # 合并函数
        def union(x, y):
            parent[find(x)] = find(y)
        
        # 存储MST的边
        mst_edges = []
        
        # Kruskal算法的主要循环
        for weight, u, v in edges:
            if find(u) != find(v):  # 如果添加(u,v)不会形成环
                union(u, v)
                mst_edges.append((u, v, weight))
        
        return mst_edges
