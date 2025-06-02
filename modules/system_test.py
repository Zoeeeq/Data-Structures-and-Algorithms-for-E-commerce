from modules.marketing_task import MarketingTask, TaskScheduler, TaskSchedulerWithDependencies
from modules.customer_network import Customer, CustomerNetwork
from modules.product_search import Product, ProductBST
from modules.disk_based_tree import DiskBasedProductIndex
import os
import traceback

# 创建目录
if not os.path.exists('visualization'):
    os.makedirs('visualization')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from modules.visualization_utils import (
    visualize_heap, 
    visualize_customer_network, 
    visualize_pagerank_convergence,
    visualize_influence_propagation,
    visualize_bplustree_search, 
    visualize_trie_search,
    visualize_performance_comparison,
    visualize_io_comparison,
    visualize_transitive_closure,
	visualize_minimum_spanning_tree
)

def test_marketing_task_scheduler():
    """测试营销任务调度功能"""
    print("=== 营销任务调度测试 ===")
    
    # 创建调度器
    scheduler = TaskScheduler()
    
    # 插入任务
    scheduler.insert_task(MarketingTask("Task A", 8, 5))   # 优先级 40
    scheduler.insert_task(MarketingTask("Task B", 6, 9))   # 优先级 54
    scheduler.insert_task(MarketingTask("Task C", 10, 10))  # 优先级 100
    scheduler.insert_task(MarketingTask("Task D", 7, 7))   # 优先级 49
    scheduler.insert_task(MarketingTask("Task E", 9, 8))   # 优先级 72
    
    print("== 所有任务 ==")
    print(scheduler)
    print("\n== 最高优先级任务 ==")
    print(scheduler.get_highest_priority_task())
    
    print("\n== 前3个最高优先级任务 ==")
    top_tasks = scheduler.get_top_k_tasks(3)
    for task in top_tasks:
        print(task)
    
    print("\n== 执行最高优先级任务 ==")
    executed = scheduler.execute_highest_priority_task()
    print(f"执行: {executed}")
    
    print("\n== 更新任务 ==")
    scheduler.update_task("Task B", 9, 10)  # 更新为优先级 90
    print(scheduler.get_highest_priority_task())
    
    print("\n== 删除任务 ==")
    scheduler.delete_task("Task B")
    print("剩余任务:")
    print(scheduler)
    
    print("\n== 测试依赖任务 ==")
    # 创建调度器
    dep_scheduler = TaskSchedulerWithDependencies()
    
    # 添加任务
    dep_scheduler.add_task(MarketingTask("Task A", 9, 8))  # 优先级 72
    dep_scheduler.add_task(MarketingTask("Task B", 7, 6))  # 优先级 42
    dep_scheduler.add_task(MarketingTask("Task C", 8, 9))  # 优先级 72
    dep_scheduler.add_task(MarketingTask("Task D", 10, 10))  # 优先级 100
    
    # 添加依赖关系: B依赖A, C依赖B, D依赖C
    dep_scheduler.add_dependency("Task B", "Task A")
    dep_scheduler.add_dependency("Task C", "Task B")
    dep_scheduler.add_dependency("Task D", "Task C")
    
    print(dep_scheduler)
    
    print("\n执行可执行任务:")
    while True:
        task = dep_scheduler.execute_highest_priority_executable_task()
        if not task:
            break
        print(f"执行: {task}")
    
    # 可视化堆结构和任务依赖关系
    heap_viz = None
    heap_viz = visualize_heap(scheduler.heap[:scheduler.size])

    return scheduler, dep_scheduler, heap_viz

def test_customer_network():
    """测试客户网络与影响力传播分析功能"""
    print("\n=== 客户网络分析测试 ===")
    
    # 创建客户网络
    network = CustomerNetwork()
    
    # 添加客户
    network.add_customer(Customer("Alice"))
    network.add_customer(Customer("Bob"))
    network.add_customer(Customer("Charlie"))
    network.add_customer(Customer("David"))
    network.add_customer(Customer("Eve"))
    
    # 添加关系
    network.add_relationship("Alice", "Bob", 0.7)
    network.add_relationship("Alice", "Charlie", 0.4)
    network.add_relationship("Bob", "David", 0.6)
    network.add_relationship("Charlie", "Eve", 0.8)
    network.add_relationship("David", "Eve", 0.3)
    network.add_relationship("Eve", "Alice", 0.2)  # 形成环
    
    print("== 客户网络 ==")
    print(network)
    
    print("\n== 客户重要性 ==")
    importance = network.calculate_customer_importance()
    for customer, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{customer}: {score:.4f}")
    
    print("\n== 客户影响范围 ==")
    for customer in network.customers:
        influenced = network.find_influenced_customers(customer)
        print(f"{customer} 可影响: {', '.join(influenced) if influenced else '无'}")
    
    print("\n== 带限制的客户影响 (最大深度=2, 最小影响力=0.3) ==")
    for customer in network.customers:
        influence_map = network.find_influenced_customers_with_limits(customer, max_depth=2, min_influence=0.3)
        if influence_map:
            influences = [f"{c}({v:.2f})" for c, v in influence_map.items()]
            print(f"{customer} 可影响: {', '.join(influences)}")
        else:
            print(f"{customer} 影响力有限")
    
    print("\n== 传递闭包 ==")
    closure = network.calculate_transitive_closure()
    for customer, reachable in closure.items():
        print(f"{customer} 可达: {', '.join(reachable) if reachable else '无'}")
    
    print("\n== 最小生成树 (Prim) ==")
    mst_prim = network.find_minimum_spanning_tree(algorithm='prim')
    for u, v, weight in mst_prim:
        print(f"{u} -- {v}: {weight:.2f}")
    
    print("\n== 最小生成树 (Kruskal) ==")
    mst_kruskal = network.find_minimum_spanning_tree(algorithm='kruskal')
    for u, v, weight in mst_kruskal:
        print(f"{u} -- {v}: {weight:.2f}")
    
    # 可视化
    network_viz = None
    pagerank_viz = None
    influence_viz = None
    
    network_viz = visualize_customer_network(network)
    pagerank_viz = visualize_pagerank_convergence(network)
    influence_viz = visualize_influence_propagation(network, "Alice", max_depth=2, min_influence=0.3)
    closure_viz = visualize_transitive_closure(network, closure)
    mst_viz = visualize_minimum_spanning_tree(network, mst_prim)

    return network, network_viz, pagerank_viz, influence_viz, closure_viz, mst_viz 

def test_product_search():
    """测试商品数据检索功能"""
    print("\n=== 商品数据检索测试 ===")
    
    # 创建商品BST
    product_bst = ProductBST()
    
    # 插入商品
    product_bst.insert(Product("Laptop", 6999, 85))
    product_bst.insert(Product("Smartphone", 3999, 90))
    product_bst.insert(Product("Earbuds", 999, 80))
    product_bst.insert(Product("Tablet", 2999, 75))
    product_bst.insert(Product("Smartwatch", 1999, 70))
    
    print("== 所有商品 ==")
    print(product_bst)
    
    print("\n== 价格范围查询 (1000-4000) ==")
    products = product_bst.find_products_in_price_range(1000, 4000)
    for product in products:
        print(product)
    
    print("\n== 更新商品 ==")
    product_bst.update("Smartphone", 3499, 95)
    print(product_bst.find_product("Smartphone"))
    
    print("\n== 删除商品 ==")
    product_bst.delete("Earbuds")
    print("剩余商品:")
    print(product_bst)
    
    return product_bst


def test_disk_based_product_index():
    """测试磁盘化商品检索功能和前缀搜索"""
    print("\n=== 磁盘化商品检索和前缀搜索测试 ===")
    
    # 创建磁盘索引
    index = DiskBasedProductIndex(order=5)
    
    # 插入商品
    products = [
        ("AppleLaptop", 8999, 92),
        ("ApplePhone", 5999, 96),
        ("SamsungPhone", 4999, 88),
        ("SonyHeadphones", 1999, 85),
        ("AppleWatch", 2999, 90),
        ("SamsungTablet", 3599, 82),
        ("SonyCamera", 6999, 79),
        ("ApplePad", 3999, 94),
        ("SamsungWatch", 1899, 83),
        ("SonyTV", 9999, 89)
    ]
    
    for name, price, popularity in products:
        index.insert(name, price, popularity)
    
    print("== 所有商品 ==")
    print(index)
    
    print("\n== 价格范围查询 (2000-5000) ==")
    for product in index.find_products_in_price_range(2000, 5000):
        print(product)
    
    print("\n== 前缀搜索 'Apple' ==")
    suggestions = index.suggest_products_by_prefix("Apple")
    for product in suggestions:
        print(product)
    
    print("\n== 前缀搜索 'Samsung' ==")
    suggestions = index.suggest_products_by_prefix("Samsung")
    for product in suggestions:
        print(product)
    
    print("\n== 更新商品 ==")
    index.update("ApplePhone", 6299, 97)
    print(index.find_product("ApplePhone"))
    
    print("\n== 删除商品后的前缀搜索 ==")
    index.delete("AppleLaptop")
    suggestions = index.suggest_products_by_prefix("Apple")
    for product in suggestions:
        print(product)
    
    # 可视化
    bplus_viz = None
    trie_viz = None
    
    bplus_viz = visualize_bplustree_search(2000, 5000)
    trie_viz = visualize_trie_search("Apple")

    return index, bplus_viz, trie_viz


def test_performance_comparison():
    """测试并可视化性能对比"""
    print("\n=== 性能比较测试 ===")
    
    perf_viz = None
    io_viz = None
    
    perf_viz = visualize_performance_comparison()
    io_viz = visualize_io_comparison()
    
    return perf_viz, io_viz


if __name__ == "__main__":
    # 营销任务调度测试
    scheduler, dep_scheduler, heap_viz = test_marketing_task_scheduler()
    
    # 客户网络分析测试
    network, network_viz, pagerank_viz, influence_viz, closure_viz, mst_viz = test_customer_network()
    
    # 商品数据检索测试
    product_bst = test_product_search()
    
    # 磁盘化商品检索和前缀搜索测试
    index, bplus_viz, trie_viz = test_disk_based_product_index()
    
    # 性能对比测试
    perf_viz, io_viz = test_performance_comparison()
