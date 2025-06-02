import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

# 创建目录
if not os.path.exists('visualization'):
    os.makedirs('visualization')

# 配色方案
COLOR_BLUE_LIGHT = '#D4E6F1'
COLOR_BLUE_MEDIUM = '#5DADE2'
COLOR_BLUE_DARK = '#2874A6'
COLOR_BLUE_ACCENT = '#1A5276'
COLOR_HIGHLIGHT = '#85C1E9'

def visualize_heap(heap, filename="heap_visualization.png"):
    """可视化最大堆结构"""
    plt.figure(figsize=(10, 6))
    
    # 创建树形结构坐标与节点关系
    G = nx.DiGraph()
    positions = {}
    labels = {}
    
    heap_size = len(heap)
    for i in range(heap_size):
        G.add_node(i)
        # 计算树状位置
        level = int(np.floor(np.log2(i + 1)))
        x = (i + 1) - 2**level + 2**(level-1)
        y = -level
        positions[i] = (x, y)
        labels[i] = f"{heap[i].name}\n{int(heap[i].priority)}"
        
        # 添加父子节点边
        left = 2*i + 1
        right = 2*i + 2
        if left < heap_size:
            G.add_edge(i, left)
        if right < heap_size:
            G.add_edge(i, right)
    
    # 绘制堆
    nx.draw(G, pos=positions, labels=labels, node_color=COLOR_BLUE_LIGHT, 
            edge_color=COLOR_BLUE_DARK, node_size=2000, arrows=False, with_labels=True)
    plt.title("Marketing Task Priority Heap")
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_customer_network(network, filename="customer_network.png"):
    """可视化客户关系网络"""
    plt.figure(figsize=(10, 8))
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    for customer in network.customers:
        G.add_node(customer)
    
    for customer, neighbors in network.graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(customer, neighbor, weight=weight)
    
    # 计算PageRank并设置节点大小
    pagerank = network.calculate_customer_importance()
    node_sizes = [pagerank[node] * 5000 for node in G.nodes()]
    
    # 获取边权重
    edge_weights = [G[u][v]['weight'] * 2.5 for u, v in G.edges()]
    
    # 绘制网络图
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=COLOR_BLUE_LIGHT)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=COLOR_BLUE_DARK, 
                          arrows=True, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G, pos)
    
    # 添加边标签（权重）
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title('Customer Network (Node Size = PageRank Value)')
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_pagerank_convergence(network, iterations=10, filename="pagerank_convergence.png"):
    """可视化PageRank收敛过程"""
    plt.figure(figsize=(10, 6))
    
    # 准备数据并计算迭代过程
    convergence_data = {customer: [] for customer in network.customers}
    num_customers = len(network.customers)
    damping = 0.85
    page_rank = {customer: 1.0 / num_customers for customer in network.graph}
    
    # 记录初始值
    for customer in convergence_data:
        convergence_data[customer].append(page_rank[customer])
    
    # 迭代计算并记录
    for i in range(iterations):
        new_rank = {customer: (1 - damping) / num_customers for customer in network.graph}
        
        for customer, neighbors in network.graph.items():
            outgoing_links = len(neighbors)
            if outgoing_links > 0:
                outgoing_sum = sum(weight for _, weight in neighbors)
                for neighbor, weight in neighbors:
                    new_rank[neighbor] += damping * page_rank[customer] * weight / outgoing_sum
        
        page_rank = new_rank
        
        for customer in convergence_data:
            convergence_data[customer].append(page_rank[customer])
    
    # 绘制收敛过程
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(convergence_data)))
    for i, (customer, values) in enumerate(convergence_data.items()):
        plt.plot(range(iterations + 1), values, marker='o', label=customer, 
                 color=blues[i], linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('PageRank Value')
    plt.title('PageRank Algorithm Convergence')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_influence_propagation(network, start_customer, max_depth=2, min_influence=0.3, filename="influence_propagation.png"):
    """可视化从特定客户出发的影响力传播"""
    plt.figure(figsize=(10, 8))
    
    # 获取影响力传播数据
    influence_map = network.find_influenced_customers_with_limits(start_customer, max_depth, min_influence)
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加起始节点
    G.add_node(start_customer)
    
    # 添加受影响的节点和边
    for customer, influence in influence_map.items():
        G.add_node(customer)
    
    # 通过BFS构建影响关系
    queue = [(start_customer, 1.0, 0)]  # (客户, 影响力, 深度)
    visited = {start_customer}
    
    while queue:
        current, current_influence, depth = queue.pop(0)
        
        if depth >= max_depth:
            continue
        
        for neighbor, weight in network.graph[current]:
            new_influence = current_influence * weight
            if new_influence >= min_influence and neighbor in influence_map:
                G.add_edge(current, neighbor, weight=new_influence)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_influence, depth + 1))
    
    # 边的属性
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    # 创建蓝色渐变效果
    edge_colors = [plt.cm.Blues(G[u][v]['weight']) for u, v in G.edges()]
    
    # 节点的属性
    node_sizes = [1200 if node == start_customer else 900 for node in G.nodes()]
    node_colors = [COLOR_BLUE_DARK if node == start_customer else COLOR_BLUE_LIGHT for node in G.nodes()]
    
    # 绘制网络图
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, 
                          arrows=True, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G, pos)
    
    # 添加边标签（影响力）
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title(f'Influence Propagation from {start_customer} (Max Depth={max_depth}, Min Influence={min_influence})')
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_bplustree_search(min_price, max_price, filename="bplus_tree_search.png"):
    """可视化B+树范围查询过程"""
    plt.figure(figsize=(14, 10))
    
    # 创建示例B+树结构
    G = nx.DiGraph()
    
    # 添加节点和边
    G.add_node("root", label="4000", node_type="internal")
    G.add_node("internal1", label="1500, 3000", node_type="internal")
    G.add_node("internal2", label="6000, 8000", node_type="internal")
    
    G.add_node("leaf1", label="999\nWireless Earbuds", node_type="leaf")
    G.add_node("leaf2", label="1999\nSmart Watch", node_type="leaf", in_range=True)
    G.add_node("leaf3", label="3499\nSmartphone", node_type="leaf", in_range=True)
    G.add_node("leaf4", label="4999\nGame Console", node_type="leaf", in_range=True)
    G.add_node("leaf5", label="6999\nLaptop", node_type="leaf")
    
    # 添加边
    G.add_edge("root", "internal1")
    G.add_edge("root", "internal2")
    G.add_edge("internal1", "leaf1")
    G.add_edge("internal1", "leaf2")
    G.add_edge("internal1", "leaf3")
    G.add_edge("internal2", "leaf4")
    G.add_edge("internal2", "leaf5")
    
    # 叶子节点链表连接
    G.add_edge("leaf1", "leaf2", link=True)
    G.add_edge("leaf2", "leaf3", link=True)
    G.add_edge("leaf3", "leaf4", link=True)
    G.add_edge("leaf4", "leaf5", link=True)
    
    # 节点位置和样式设置
    pos = {
        "root": (6, 3),
        "internal1": (3, 2),
        "internal2": (9, 2),
        "leaf1": (1, 1),
        "leaf2": (3, 1),
        "leaf3": (5, 1),
        "leaf4": (7, 1),
        "leaf5": (9, 1)
    }
    
    # 设置节点颜色
    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get('node_type') == "internal":
            node_colors.append(COLOR_BLUE_LIGHT)
        elif data.get('in_range', False):
            node_colors.append(COLOR_HIGHLIGHT)  # 在查询范围内的节点
        else:
            node_colors.append('#E5E8E8')  # 浅灰色
    
    # 绘制B+树
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if not d.get('link', False)], 
                          arrows=True, edge_color=COLOR_BLUE_DARK)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get('link', False)], 
                           style='dashed', edge_color=COLOR_BLUE_DARK, arrows=True)
    
    # 获取节点标签并绘制
    labels = {node: data.get('label', '') for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    # 查询路径高亮
    path_edges = [("root", "internal1"), ("internal1", "leaf2")]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, arrows=True, width=2.5, 
                          edge_color='#3498DB', arrowstyle='->')
    
    # 范围扫描路径
    scan_path = [("leaf2", "leaf3"), ("leaf3", "leaf4")]
    nx.draw_networkx_edges(G, pos, edgelist=scan_path, arrows=True, width=2.0, 
                          edge_color='#2ECC71', style='dashed', arrowstyle='->')
    
    # 添加图例和标题信息
    plt.text(3, 0, f"Query Range: {min_price} - {max_price}", fontsize=12, 
             bbox=dict(facecolor='#F4F6F7', alpha=0.8, edgecolor=COLOR_BLUE_DARK))
    plt.title("B+ Tree Range Query", fontsize=16, fontweight='bold')
    
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_trie_search(prefix, filename="trie_search.png"):
    """可视化前缀树搜索过程"""
    plt.figure(figsize=(12, 10))
    
    # 创建示例前缀树
    G = nx.DiGraph()
    
    # 添加节点和边
    G.add_node("root", label="root")
    G.add_node("A", label="A")
    G.add_node("S", label="S")
    G.add_edge("root", "A")
    G.add_edge("root", "S")
    
    # Apple路径
    G.add_node("A_p", label="p")
    G.add_node("A_p_p", label="p")
    G.add_node("A_p_p_l", label="l")
    G.add_node("A_p_p_l_e", label="e")
    G.add_edge("A", "A_p")
    G.add_edge("A_p", "A_p_p")
    G.add_edge("A_p_p", "A_p_p_l")
    G.add_edge("A_p_p_l", "A_p_p_l_e")
    
    # 产品节点
    G.add_node("ApplePhone", label="ApplePhone\nPopularity:96")
    G.add_node("ApplePad", label="ApplePad\nPopularity:94")
    G.add_node("AppleLaptop", label="AppleLaptop\nPopularity:92")
    G.add_node("AppleWatch", label="AppleWatch\nPopularity:90")
    G.add_edge("A_p_p_l_e", "ApplePhone")
    G.add_edge("A_p_p_l_e", "ApplePad")
    G.add_edge("A_p_p_l_e", "AppleLaptop")
    G.add_edge("A_p_p_l_e", "AppleWatch")
    
    # 决定搜索路径上的节点
    search_nodes = ["root", "A", "A_p", "A_p_p", "A_p_p_l", "A_p_p_l_e"]
    search_edges = list(zip(search_nodes[:-1], search_nodes[1:]))
    
    # 设置节点位置 - 使用自定义布局替代pygraphviz
    # 手动定义每个节点的位置
    pos = {
        "root": (5, 10),
        "A": (3, 8),
        "S": (7, 8),
        "A_p": (3, 6),
        "A_p_p": (3, 4),
        "A_p_p_l": (3, 2),
        "A_p_p_l_e": (3, 0),
        "ApplePhone": (1, -2),
        "ApplePad": (3, -2),
        "AppleLaptop": (5, -2),
        "AppleWatch": (7, -2)
    }
    
    # 设置节点和边的颜色
    node_colors = [COLOR_HIGHLIGHT if node in search_nodes else
                  '#AED6F1' if node in ["ApplePhone", "ApplePad", "AppleLaptop", "AppleWatch"] else
                  COLOR_BLUE_LIGHT for node in G.nodes()]
    
    edge_colors = [COLOR_BLUE_DARK if (u, v) in search_edges else '#A9CCE3' for u, v in G.edges()]
    
    # 绘制前缀树
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, 
                          arrowstyle='->', arrowsize=15)
    
    # 绘制节点标签
    labels = {node: data.get('label', '') for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    # 添加搜索信息
    plt.title(f"Prefix Tree Search: '{prefix}'", fontsize=16, fontweight='bold')
    
    # 修改表格位置到右上角，避免遮挡树图
    table_data = [
        ["ApplePhone", "96", "5999"],
        ["ApplePad", "94", "3999"], 
        ["AppleLaptop", "92", "8999"],
        ["AppleWatch", "90", "2999"]
    ]
    
    # 将表格移到右上角位置
    table_ax = plt.axes([0.65, 0.35, 0.3, 0.2])  # 从[0.65, 0.10, 0.3, 0.2]改为[0.65, 0.65, 0.3, 0.2]
    table_ax.axis('off')
    table = plt.table(cellText=table_data, 
                    colLabels=['Product', 'Popularity', 'Price'],
                    loc='center',
                    cellLoc='center',
                    cellColours=[[COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT]] * len(table_data),
                    colColours=[COLOR_BLUE_MEDIUM] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.text(0.5, 1.1, "Search Results (Sorted by Popularity)", 
            horizontalalignment='center', transform=table_ax.transAxes)
    
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_performance_comparison(filename="performance_comparison.png"):
    """可视化不同数据结构的性能对比"""
    plt.figure(figsize=(12, 8))
    
    # 操作类型
    operations = ['Insert', 'Delete', 'Update', 'Single Query', 'Range Query']
    
    # 不同数据结构的性能数据 (模拟的相对性能)
    bst_data = [12, 12, 12, 12, 40]
    bplus_data = [8, 8, 8, 8, 12]
    heap_data = [9, 9, 9, 3, 48]
    
    x = np.arange(len(operations))  # 操作位置
    width = 0.25  # 柱状图宽度
    
    # 使用蓝色系
    bst_color = '#5DADE2'  # 浅蓝
    bplus_color = '#2874A6'  # 中蓝
    heap_color = '#1A5276'  # 深蓝
    
    # 绘制柱状图
    plt.bar(x - width, bst_data, width, label='Binary Search Tree', color=bst_color)
    plt.bar(x, bplus_data, width, label='B+ Tree', color=bplus_color)
    plt.bar(x + width, heap_data, width, label='Max Heap', color=heap_color)
    
    # 添加标签和标题
    plt.xlabel('Operation Type', fontsize=12)
    plt.ylabel('Relative Performance (ms, lower is better)', fontsize=12)
    plt.title('Data Structure Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, operations)
    plt.legend()
    
    # 添加值标签
    for i, v in enumerate(bst_data):
        plt.text(i - width, v + 1, str(v), ha='center')
    for i, v in enumerate(bplus_data):
        plt.text(i, v + 1, str(v), ha='center')
    for i, v in enumerate(heap_data):
        plt.text(i + width, v + 1, str(v), ha='center')
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_io_comparison(filename="io_comparison.png"):
    """可视化磁盘I/O次数对比"""
    plt.figure(figsize=(10, 6))
    
    # 操作类型和数据规模
    operations = ['Single Query', 'Range Query\n(10 items)', 'Range Query\n(100 items)', 'Range Query\n(1000 items)']
    
    # 不同数据结构的I/O次数 (模拟数据)
    bst_io = [12, 22, 112, 1012]
    bplus_io = [3, 4, 6, 12]
    
    x = np.arange(len(operations))  # 操作位置
    width = 0.35  # 柱状图宽度
    
    # 使用蓝色系
    bst_color = '#5DADE2'  # 浅蓝
    bplus_color = '#2874A6'  # 中蓝
    
    # 绘制柱状图
    plt.bar(x - width/2, bst_io, width, label='Binary Search Tree', color=bst_color)
    plt.bar(x + width/2, bplus_io, width, label='B+ Tree', color=bplus_color)
    
    # 对数比例显示
    plt.yscale('log')
    
    # 添加标签和标题
    plt.xlabel('Operation Type', fontsize=12)
    plt.ylabel('Disk I/O Count (log scale)', fontsize=12)
    plt.title('B+ Tree vs Binary Search Tree Disk I/O Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, operations)
    plt.legend()
    
    # 添加值标签
    for i, v in enumerate(bst_io):
        plt.text(i - width/2, v * 1.1, str(v), ha='center')
    for i, v in enumerate(bplus_io):
        plt.text(i + width/2, v * 1.1, str(v), ha='center')
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_transitive_closure(network, closure, filename="transitive_closure.png"):
    """可视化传递闭包"""
    plt.figure(figsize=(12, 10))
    
    G = nx.DiGraph()
    
    for customer in network.customers:
        G.add_node(customer)
    
    for customer, neighbors in network.graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(customer, neighbor, weight=weight, original=True)

    for customer, reachable in closure.items():
        for neighbor in reachable:
            if not G.has_edge(customer, neighbor):
                G.add_edge(customer, neighbor, transitive=True)
    
    pos = nx.spring_layout(G, seed=42)
    
    original_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('original', False)]
    nx.draw_networkx_edges(G, pos, edgelist=original_edges, 
                         width=2, edge_color=COLOR_BLUE_DARK, 
                         arrows=True, arrowstyle='->', arrowsize=15)
    
    transitive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('transitive', False)]
    nx.draw_networkx_edges(G, pos, edgelist=transitive_edges, 
                         width=1, edge_color=COLOR_BLUE_LIGHT, style='dashed',
                         arrows=True, arrowstyle='->', arrowsize=10)
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=COLOR_BLUE_MEDIUM)
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Customer Network Transitive Closure (Solid=Original, Dashed=Transitive)')
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"

def visualize_minimum_spanning_tree(network, mst_edges, filename="minimum_spanning_tree.png"):
    """可视化最小生成树"""
    plt.figure(figsize=(12, 10))
    
    G_original = nx.Graph()
    
    for customer in network.customers:
        G_original.add_node(customer)
    
    undirected_edges = {}
    for customer, neighbors in network.graph.items():
        for neighbor, weight in neighbors:
            edge = tuple(sorted([customer, neighbor]))
            if edge in undirected_edges:
                undirected_edges[edge] = (undirected_edges[edge] + weight) / 2
            else:
                undirected_edges[edge] = weight
    
    for (u, v), weight in undirected_edges.items():
        G_original.add_edge(u, v, weight=weight)
    
    G_mst = nx.Graph()
    for customer in network.customers:
        G_mst.add_node(customer)
    
    for u, v, weight in mst_edges:
        G_mst.add_edge(u, v, weight=weight)
    
    pos = nx.spring_layout(G_original, seed=42)
    
    nx.draw_networkx_edges(G_original, pos, width=1, alpha=0.3, edge_color='gray')
    nx.draw_networkx_edges(G_mst, pos, width=3, edge_color=COLOR_BLUE_DARK)
    
    edge_labels = {(u, v): f"{G_mst[u][v]['weight']:.2f}" for u, v in G_mst.edges()}
    nx.draw_networkx_edge_labels(G_mst, pos, edge_labels=edge_labels, font_size=9)
    
    nx.draw_networkx_nodes(G_original, pos, node_size=700, node_color=COLOR_BLUE_MEDIUM)
    nx.draw_networkx_labels(G_original, pos)
    
    total_weight = sum(data['weight'] for _, _, data in G_mst.edges(data=True))
    
    plt.title(f'Customer Network Minimum Spanning Tree (Total Influence Cost: {total_weight:.2f})')
    plt.axis('off')
    plt.savefig(f"visualization/{filename}")
    plt.close()
    
    return f"visualization/{filename}"