class DiskProduct:
    """表示磁盘上的商品记录"""
    
    def __init__(self, name, price, popularity, disk_pos=None):
        """
        初始化一个商品记录
        
        参数:
            name (str): 商品名称
            price (float): 商品价格
            popularity (float): 商品热度
            disk_pos (int, optional): 商品在磁盘上的位置
        """
        self.name = name
        self.price = price
        self.popularity = popularity
        self.disk_pos = disk_pos
    
    def __str__(self):
        """商品记录"""
        return f"商品: {self.name}, 价格: {self.price}, 热度: {self.popularity}, 磁盘位置: {self.disk_pos}"


class BPlusTreeNode:
    """B+树节点"""
    
    def __init__(self, is_leaf=True, order=5):
        """
        初始化B+树节点
        
        参数:
            is_leaf (bool): 是否是叶子节点
            order (int): B+树的阶数（最大子节点数）
        """
        self.is_leaf = is_leaf
        self.order = order
        self.keys = []         # 存储键(价格)
        self.children = []     # 存储子节点或商品记录
        self.next_leaf = None  # 指向下一个叶子节点（只在叶子节点中使用）
    
    def is_full(self):
        """检查节点是否已满"""
        return len(self.keys) >= self.order - 1


class BPlusTree:
    """
    B+树实现的商品索引
    主键为商品价格，支持范围查询和商品记录的管理
    """
    
    def __init__(self, order=5):
        """
        初始化B+树
        
        参数:
            order (int): B+树的阶数
        """
        self.root = BPlusTreeNode(is_leaf=True, order=order)
        self.order = order
        
        # 商品名称到商品记录的映射
        self.product_map = {}
    
    def _find_leaf(self, key):
        """
        找到应该包含指定键的叶子节点
        
        参数:
            key (float): 要查找的键值（价格）
            
        返回:
            BPlusTreeNode: 包含key的叶子节点
        """
        node = self.root
        
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        
        return node
    
    def insert(self, product):
        """
        插入新商品
        
        参数:
            product (DiskProduct): 要插入的商品
            
        返回:
            bool: 成功返回True
        """
        # 如果商品已存在，更新它
        if product.name in self.product_map:
            self.update(product.name, product.price, product.popularity)
            return True
        
        # 记录商品
        self.product_map[product.name] = product
        
        # 查找要插入的叶子节点
        key = product.price
        leaf = self._find_leaf(key)
        
        # 在叶子节点中找到正确的插入位置
        i = 0
        while i < len(leaf.keys) and key > leaf.keys[i]:
            i += 1
        
        # 插入键和值
        leaf.keys.insert(i, key)
        leaf.children.insert(i, product)
        
        # 如果节点溢出，分裂节点
        if leaf.is_full():
            self._split_leaf(leaf)
        
        return True
    
    def _split_leaf(self, leaf):
        """
        分裂叶子节点
        
        参数:
            leaf (BPlusTreeNode): 要分裂的叶子节点
        """
        # 创建新叶子节点
        new_leaf = BPlusTreeNode(is_leaf=True, order=self.order)
        
        # 确定分裂点
        mid = self.order // 2
        
        # 将右半部分移动到新节点
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.children = leaf.children[mid:]
        
        # 更新原节点
        leaf.keys = leaf.keys[:mid]
        leaf.children = leaf.children[:mid]
        
        # 维护叶子节点链表
        new_leaf.next_leaf = leaf.next_leaf
        leaf.next_leaf = new_leaf
        
        # 将中间键插入父节点
        self._insert_in_parent(leaf, new_leaf.keys[0], new_leaf)
    
    def _insert_in_parent(self, left, key, right):
        """
        在父节点中插入键和右子节点
        
        参数:
            left (BPlusTreeNode): 左子节点
            key (float): 要插入的键
            right (BPlusTreeNode): 右子节点
        """
        # 如果left是根节点，创建新的根
        if left == self.root:
            new_root = BPlusTreeNode(is_leaf=False, order=self.order)
            new_root.keys = [key]
            new_root.children = [left, right]
            self.root = new_root
            return
        
        # 找到父节点
        parent = self._find_parent(self.root, left)
        
        # 在父节点中插入key和right
        i = 0
        while i < len(parent.keys) and key > parent.keys[i]:
            i += 1
        
        parent.keys.insert(i, key)
        parent.children.insert(i + 1, right)
        
        # 如果父节点溢出，分裂内部节点
        if parent.is_full():
            self._split_internal(parent)
    
    def _find_parent(self, node, child):
        """
        递归查找子节点的父节点
        
        参数:
            node (BPlusTreeNode): 当前节点
            child (BPlusTreeNode): 要查找的子节点
            
        返回:
            BPlusTreeNode: 父节点或None
        """
        if node.is_leaf or any(c.is_leaf for c in node.children):
            for i, c in enumerate(node.children):
                if c == child:
                    return node
            return None
        
        # 递归搜索可能包含child的子树
        for c in node.children:
            if result := self._find_parent(c, child):
                return result
        
        return None
    
    def _split_internal(self, node):
        """
        分裂内部节点
        
        参数:
            node (BPlusTreeNode): 要分裂的内部节点
        """
        # 创建新内部节点
        new_node = BPlusTreeNode(is_leaf=False, order=self.order)
        
        # 确定分裂点
        mid = self.order // 2
        
        # 将右半部分移动到新节点
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        
        # 获取中间键并更新原节点
        middle_key = node.keys[mid]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        
        # 将中间键插入父节点
        self._insert_in_parent(node, middle_key, new_node)
    
    def delete(self, name):
        """
        删除商品
        
        参数:
            name (str): 要删除的商品名称
            
        返回:
            bool: 成功返回True，商品不存在返回False
        """
        if name not in self.product_map:
            return False
        
        product = self.product_map[name]
        key = product.price
        
        # 查找包含此商品的叶子节点
        leaf = self._find_leaf(key)
        
        # 在叶子节点中查找商品
        found = False
        for i, child in enumerate(leaf.children):
            if isinstance(child, DiskProduct) and child.name == name:
                # 删除键和值
                leaf.keys.pop(i)
                leaf.children.pop(i)
                found = True
                break
        
        if not found:
            return False
        
        # 从映射中删除商品
        del self.product_map[name]
        
        return True
    
    def update(self, name, price=None, popularity=None):
        """
        更新商品信息
        
        参数:
            name (str): 要更新的商品名称
            price (float, optional): 新价格
            popularity (float, optional): 新热度
            
        返回:
            bool: 成功返回True，商品不存在返回False
        """
        if name not in self.product_map:
            return False
        
        product = self.product_map[name]
        old_price = product.price
        
        # 如果价格没变，直接更新商品记录
        if price is None or price == old_price:
            if popularity is not None:
                product.popularity = popularity
            return True
        
        # 价格变了，需要重新插入
        new_product = DiskProduct(name, price if price is not None else old_price,
                                popularity if popularity is not None else product.popularity,
                                product.disk_pos)
        
        # 先删除旧记录
        self.delete(name)
        
        # 插入新商品
        self.insert(new_product)
        return True
    
    def find_product(self, name):
        """
        根据名称查找商品
        
        参数:
            name (str): 商品名称
            
        返回:
            DiskProduct: 找到的商品，如果不存在则返回None
        """
        if name in self.product_map:
            return self.product_map[name]
        return None
    
    def find_products_in_price_range(self, min_price, max_price):
        """
        查找指定价格范围内的所有商品
        
        参数:
            min_price (float): 最小价格
            max_price (float): 最大价格
            
        返回:
            list: 符合条件的商品列表
        """
        result = []
        
        # 查找最小价格对应的叶子节点
        leaf = self._find_leaf(min_price)
        
        # 遍历叶子节点链表
        while leaf:
            # 收集当前叶子节点中满足条件的商品
            for i, key in enumerate(leaf.keys):
                if min_price <= key <= max_price:
                    result.append(leaf.children[i])
                elif key > max_price:
                    # 跳出内层循环
                    break
            
            # 如果最后一个键大于最大价格，退出
            if leaf.keys and leaf.keys[-1] > max_price:
                break
            
            # 移动到下一个叶子节点
            leaf = leaf.next_leaf
        
        return result


class TrieNode:
    """前缀树节点，用于商品名称前缀搜索"""
    
    def __init__(self):
        """初始化前缀树节点"""
        self.children = {}  # 字符到子节点的映射
        self.products = []  # 以当前前缀开头的商品列表
    
    def insert(self, product, index=0):
        """
        插入商品到前缀树
        
        参数:
            product (DiskProduct): 要插入的商品
            index (int): 当前处理的字符索引
        """
        # 在当前节点记录商品
        if product not in self.products:
            self.products.append(product)
            # 按热度排序，热度高的排在前面
            self.products.sort(key=lambda x: x.popularity, reverse=True)
        
        # 如果已处理完所有字符，结束递归
        if index >= len(product.name):
            return
        
        char = product.name[index]
        
        # 如果当前字符不存在对应的子节点，创建一个
        if char not in self.children:
            self.children[char] = TrieNode()
        
        # 递归处理下一个字符
        self.children[char].insert(product, index + 1)
    
    def delete(self, product, index=0):
        """
        从前缀树中删除商品
        
        参数:
            product (DiskProduct): 要删除的商品
            index (int): 当前处理的字符索引
            
        返回:
            bool: 如果该节点可以被删除（没有子节点且没有保存商品）则返回True
        """
        # 从当前节点的商品列表中移除
        if product in self.products:
            self.products.remove(product)
        
        # 如果已处理完所有字符，结束递归
        if index >= len(product.name):
            return not self.products and not self.children
        
        char = product.name[index]
        
        # 如果当前字符存在对应的子节点，递归删除
        if char in self.children:
            if self.children[char].delete(product, index + 1):
                # 如果子节点可以被删除，则删除它
                del self.children[char]
        
        # 如果该节点没有子节点且没有保存商品，可以被删除
        return not self.products and not self.children
    
    def search(self, prefix, index=0):
        """
        查找以指定前缀开头的商品
        
        参数:
            prefix (str): 前缀
            index (int): 当前处理的字符索引
            
        返回:
            list: 热度排序的商品列表
        """
        # 如果已处理完所有字符，返回该节点的商品列表
        if index >= len(prefix):
            return self.products
        
        char = prefix[index]
        
        # 如果当前字符不存在对应的子节点，返回空列表
        if char not in self.children:
            return []
        
        # 递归处理下一个字符
        return self.children[char].search(prefix, index + 1)


class DiskBasedProductIndex:
    """
    基于磁盘的商品索引系统
    使用B+树进行价格索引，使用前缀树进行名称前缀搜索
    """
    
    def __init__(self, order=5):
        """
        初始化商品索引系统
        
        参数:
            order (int): B+树的阶数
        """
        self.price_index = BPlusTree(order)  # 价格索引
        self.name_trie = TrieNode()          # 名称前缀树
        
        # 模拟磁盘存储的当前位置
        self.current_disk_pos = 0
    
    def insert(self, name, price, popularity):
        """
        插入新商品
        
        参数:
            name (str): 商品名称
            price (float): 商品价格
            popularity (float): 商品热度
            
        返回:
            bool: 成功返回True
        """
        # 分配磁盘位置
        disk_pos = self.current_disk_pos
        self.current_disk_pos += 1
        
        # 创建商品记录
        product = DiskProduct(name, price, popularity, disk_pos)
        
        # 更新索引
        self.price_index.insert(product)
        self.name_trie.insert(product)
        
        return True
    
    def delete(self, name):
        """
        删除商品
        
        参数:
            name (str): 要删除的商品名称
            
        返回:
            bool: 成功返回True，商品不存在返回False
        """
        # 查找商品
        product = self.price_index.find_product(name)
        if not product:
            return False
        
        # 从索引中删除
        self.price_index.delete(name)
        self.name_trie.delete(product)
        
        return True
    
    def update(self, name, price=None, popularity=None):
        """
        更新商品信息
        
        参数:
            name (str): 要更新的商品名称
            price (float): 新价格
            popularity (float): 新热度
            
        返回:
            bool: 成功返回True，商品不存在返回False
        """
        # 查找商品
        product = self.price_index.find_product(name)
        if not product:
            return False
        
        # 从索引中删除
        self.price_index.delete(name)
        self.name_trie.delete(product)
        
        # 更新商品属性
        new_price = price if price is not None else product.price
        new_popularity = popularity if popularity is not None else product.popularity
        
        # 重新插入更新后的商品
        self.insert(name, new_price, new_popularity)
        
        return True
    
    def find_product(self, name):
        """
        根据名称查找商品
        
        参数:
            name (str): 商品名称
            
        返回:
            DiskProduct: 找到的商品，如果不存在则返回None
        """
        return self.price_index.find_product(name)
    
    def find_products_in_price_range(self, min_price, max_price):
        """
        查找指定价格范围内的所有商品
        
        参数:
            min_price (float): 最小价格
            max_price (float): 最大价格
            
        返回:
            list: 符合条件的商品列表
        """
        return self.price_index.find_products_in_price_range(min_price, max_price)
    
    def suggest_products_by_prefix(self, prefix, limit=5):
        """
        根据名称前缀推荐热度高的商品
        
        参数:
            prefix (str): 商品名称的前缀
            limit (int): 最多返回的商品数量
            
        返回:
            list: 按热度排序的商品推荐列表
        """
        products = self.name_trie.search(prefix)
        return products[:limit]
    
    def __str__(self):
        """索引系统的字符串表示"""
        # 获取所有商品
        products = []
        for name, product in self.price_index.product_map.items():
            products.append(product)
        
        # 按价格排序
        products.sort(key=lambda x: x.price)
        
        if not products:
            return "商品索引为空"
        
        return "\n".join(str(product) for product in products)
