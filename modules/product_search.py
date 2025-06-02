class Product:
    """
    表示一个商品
    包含名称、价格和热度
    """
    
    def __init__(self, name, price, popularity):
        """
        初始化一个商品
        
        参数:
            name (str): 商品名称
            price (float): 商品价格
            popularity (float): 商品热度
        """
        self.name = name
        self.price = price
        self.popularity = popularity
    
    def __str__(self):
        """商品的字符串表示"""
        return f"商品: {self.name}, 价格: {self.price}, 热度: {self.popularity}"


class BSTNode:
    """二叉搜索树节点"""
    
    def __init__(self, product):
        """
        初始化一个节点
        
        参数:
            product (Product): 节点包含的商品
        """
        self.product = product
        self.left = None
        self.right = None
    
    def __str__(self):
        """节点的字符串表示"""
        return str(self.product)


class ProductBST:
    """
    使用二叉搜索树管理商品数据
    支持插入、删除、更新和价格范围查询
    """
    
    def __init__(self):
        """初始化空的商品BST"""
        self.root = None
        self.product_map = {}  # 商品名称到节点的映射
    
    def _insert_node(self, node, product):
        """
        递归插入节点
        
        参数:
            node (BSTNode): 当前节点
            product (Product): 要插入的商品
            
        返回:
            BSTNode: 更新后的子树根节点
        """
        if node is None:
            node = BSTNode(product)
            self.product_map[product.name] = node
            return node
        
        if product.price < node.product.price:
            node.left = self._insert_node(node.left, product)
        elif product.price > node.product.price:
            node.right = self._insert_node(node.right, product)
        else:
            # 相同价格的商品按名称字典序排序
            if product.name < node.product.name:
                node.left = self._insert_node(node.left, product)
            elif product.name > node.product.name:
                node.right = self._insert_node(node.right, product)
            else:
                # 相同名称和价格的商品，更新节点
                node.product = product
                self.product_map[product.name] = node
        
        return node
    
    def insert(self, product):
        """
        插入新商品
        
        参数:
            product (Product): 要插入的商品
            
        返回:
            bool: 成功返回True
        """
        if product.name in self.product_map:
            # 如果商品已存在，更新它
            self.update(product.name, product.price, product.popularity)
            return True
        
        self.root = self._insert_node(self.root, product)
        return True
    
    def _find_min_node(self, node):
        """
        找到子树中价格最小的节点
        
        参数:
            node (BSTNode): 子树根节点
            
        返回:
            BSTNode: 价格最小的节点
        """
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def _delete_node(self, node, name):
        """
        递归删除节点
        
        参数:
            node (BSTNode): 当前节点
            name (str): 要删除的商品名称
            
        返回:
            BSTNode: 更新后的子树根节点
        """
        if node is None:
            return None
        
        if name == node.product.name:
            # 情况1: 叶节点
            if node.left is None and node.right is None:
                del self.product_map[name]
                return None
            
            # 情况2: 只有一个子节点
            if node.left is None:
                del self.product_map[name]
                return node.right
            if node.right is None:
                del self.product_map[name]
                return node.left
            
            # 情况3: 有两个子节点
            # 找到右子树中的最小节点
            successor = self._find_min_node(node.right)
            node.product = successor.product
            self.product_map[node.product.name] = node
            
            # 删除右子树中的后继节点
            node.right = self._delete_node(node.right, successor.product.name)
        
        # 继续递归查找要删除的节点
        elif name in self.product_map:
            target_node = self.product_map[name]
            target_price = target_node.product.price
            
            if target_price < node.product.price:
                node.left = self._delete_node(node.left, name)
            else:
                node.right = self._delete_node(node.right, name)
        
        return node
    
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
        
        self.root = self._delete_node(self.root, name)
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
        if name not in self.product_map:
            return False
        
        node = self.product_map[name]
        old_price = node.product.price
        
        # 如果价格没变，直接更新节点
        if price is None or price == old_price:
            if popularity is not None:
                node.product.popularity = popularity
            return True
        
        # 价格变了，需要重新插入
        product = Product(name, price if price is not None else old_price,
                        popularity if popularity is not None else node.product.popularity)
        
        # 先删除旧节点
        self.delete(name)
        
        # 插入新商品
        self.insert(product)
        return True
    
    def find_product(self, name):
        """
        根据名称查找商品
        
        参数:
            name (str): 商品名称
            
        返回:
            Product: 找到的商品，如果不存在则返回None
        """
        if name in self.product_map:
            return self.product_map[name].product
        return None
    
    def _find_products_in_range(self, node, min_price, max_price, result):
        """
        递归查找价格范围内的所有商品
        
        参数:
            node (BSTNode): 当前节点
            min_price (float): 最小价格
            max_price (float): 最大价格
            result (list): 结果列表
        """
        if node is None:
            return
        
        # 如果当前节点价格小于最小价格，只需搜索右子树
        if node.product.price < min_price:
            self._find_products_in_range(node.right, min_price, max_price, result)
        
        # 如果当前节点价格大于最大价格，只需搜索左子树
        elif node.product.price > max_price:
            self._find_products_in_range(node.left, min_price, max_price, result)
        
        # 如果在范围内，搜索整个树并添加当前节点
        else:
            self._find_products_in_range(node.left, min_price, max_price, result)
            result.append(node.product)
            self._find_products_in_range(node.right, min_price, max_price, result)
    
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
        self._find_products_in_range(self.root, min_price, max_price, result)
        return result
    
    def __str__(self):
        """商品BST的字符串表示"""
        products = self._inorder_traversal(self.root)
        if not products:
            return "商品树为空"
        
        return "\n".join(str(product) for product in products)
    
    def _inorder_traversal(self, node):
        """
        中序遍历树
        
        参数:
            node (BSTNode): 当前节点
            
        返回:
            list: 排序后的商品列表
        """
        if node is None:
            return []
        
        return (self._inorder_traversal(node.left) + 
                [node.product] + 
                self._inorder_traversal(node.right))
