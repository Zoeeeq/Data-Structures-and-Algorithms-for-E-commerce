class MarketingTask:
    """
    表示一个营销任务
    包含任务名称、紧急度和影响力
    
    """
    
    def __init__(self, name, urgency, influence):
        """
        初始化一个营销任务
        
        参数:
            name (str): 任务名称
            urgency (float): 任务紧急度
            influence (float): 任务影响力
        """
        self.name = name
        self.urgency = urgency
        self.influence = influence
        self.priority = urgency * influence  # 计算优先级
    
    def update(self, urgency=None, influence=None):
        """
        更新任务属性并重新计算优先级
        
        参数:
            urgency (float): 新的紧急度
            influence (float): 新的影响力
        """
        if urgency is not None:
            self.urgency = urgency
        if influence is not None:
            self.influence = influence
        self.priority = self.urgency * self.influence  # 重新计算优先级
    
    def __lt__(self, other):
        """
        比较任务优先级，用于堆操作
        反向比较以构建最大堆
        """
        return self.priority > other.priority
    
    def __str__(self):
        """任务的字符串表示"""
        return f"任务: {self.name}, 优先级: {self.priority} (紧急度: {self.urgency}, 影响力: {self.influence})"


class TaskScheduler:
    """
    使用最大堆实现的营销任务优先级队列
    支持插入、删除、更新和基于优先级的操作
    """
    
    def __init__(self):
        """初始化空的任务调度器"""
        self.heap = []  # 用堆实现的优先队列
        self.task_map = {}  # 任务名称到堆中索引的映射
        self.size = 0
    
    def _swap(self, i, j):
        """
        交换堆中的两个任务并更新它们在映射中的索引
        
        参数:
            i (int): 第一个任务的索引
            j (int): 第二个任务的索引
        """
        self.task_map[self.heap[i].name] = j
        self.task_map[self.heap[j].name] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _bubble_up(self, index):
        """
        上浮操作，维护堆性质
        
        参数:
            index (int): 需要上浮的任务索引
        """
        parent = (index - 1) // 2
        if index > 0 and self.heap[index].priority > self.heap[parent].priority:
            self._swap(index, parent)
            self._bubble_up(parent)
    
    def _bubble_down(self, index):
        """
        下沉操作，维护堆性质
        
        参数:
            index (int): 需要下沉的任务索引
        """
        largest = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < self.size and self.heap[left].priority > self.heap[largest].priority:
            largest = left
        
        if right < self.size and self.heap[right].priority > self.heap[largest].priority:
            largest = right
        
        if largest != index:
            self._swap(index, largest)
            self._bubble_down(largest)
    
    def insert_task(self, task):
        """
        向调度器中插入新任务
        
        参数:
            task (MarketingTask): 要插入的任务
            
        返回:
            bool: 成功返回True
        """
        # 将任务添加到堆的末尾
        if self.size < len(self.heap):
            self.heap[self.size] = task
        else:
            self.heap.append(task)
        
        # 更新映射并上浮
        self.task_map[task.name] = self.size
        self.size += 1
        self._bubble_up(self.size - 1)
        return True
    
    def delete_task(self, task_name):
        """
        从调度器中删除任务
        
        参数:
            task_name (str): 要删除的任务名称
            
        返回:
            bool: 成功返回True，任务不存在返回False
        """
        if task_name not in self.task_map:
            return False
        
        index = self.task_map[task_name]
        
        # 如果不是最后一个任务，与最后一个任务交换
        if index < self.size - 1:
            self._swap(index, self.size - 1)
        
        # 从映射中删除并减小大小
        del self.task_map[task_name]
        self.size -= 1
        
        # 如果需要，恢复堆性质
        if index < self.size:
            # 尝试上浮和下沉（只有一个会生效）
            self._bubble_up(index)
            self._bubble_down(index)
        
        return True
    
    def update_task(self, task_name, urgency=None, influence=None):
        """
        更新任务属性
        
        参数:
            task_name (str): 要更新的任务名称
            urgency (float): 新的紧急度值
            influence (float): 新的影响力值
            
        返回:
            bool: 成功返回True，任务不存在返回False
        """
        if task_name not in self.task_map:
            return False
        
        index = self.task_map[task_name]
        self.heap[index].update(urgency, influence)
        
        # 恢复堆性质
        self._bubble_up(index)
        self._bubble_down(index)
        
        return True
    
    def get_highest_priority_task(self):
        """
        获取最高优先级的任务，不移除
        
        返回:
            MarketingTask: 最高优先级的任务，如果为空则返回None
        """
        if self.size == 0:
            return None
        return self.heap[0]
    
    def execute_highest_priority_task(self):
        """
        移除并返回最高优先级的任务
        
        返回:
            MarketingTask: 最高优先级的任务，如果为空则返回None
        """
        if self.size == 0:
            return None
        
        highest_task = self.heap[0]
        self.delete_task(highest_task.name)
        return highest_task
    
    def get_top_k_tasks(self, k):
        """
        获取前k个最高优先级的任务（不移除）
        
        参数:
            k (int): 要返回的任务数量
            
        返回:
            list: 包含最多k个最高优先级任务的列表
        """
        if self.size == 0:
            return []
        
        # 创建任务副本进行排序
        tasks = self.heap[:self.size]
        tasks.sort(key=lambda x: x.priority, reverse=True)
        return tasks[:min(k, self.size)]
    
    def __len__(self):
        """返回调度器中的任务数量"""
        return self.size
    
    def __str__(self):
        """调度器的字符串表示"""
        if self.size == 0:
            return "调度器为空"
        
        tasks = self.heap[:self.size]
        tasks.sort(key=lambda x: x.priority, reverse=True)
        return "\n".join(str(task) for task in tasks)


# 考虑任务依赖
class TaskSchedulerWithDependencies:
    """
    考虑任务之间的依赖关系
    只有在所有前置任务完成后才能执行任务
    """
    
    def __init__(self):
        """初始化支持依赖关系的调度器"""
        self.scheduler = TaskScheduler()
        self.dependencies = {}  # 任务名称到前置任务集合的映射
        self.reverse_deps = {}  # 任务名称到依赖它的任务集合的映射
    
    def add_task(self, task):
        """
        添加新任务到调度器
        
        参数:
            task (MarketingTask): 要添加的任务
            
        返回:
            bool: 成功返回True
        """
        result = self.scheduler.insert_task(task)
        if task.name not in self.dependencies:
            self.dependencies[task.name] = set()
        if task.name not in self.reverse_deps:
            self.reverse_deps[task.name] = set()
        return result
    
    def add_dependency(self, task_name, prerequisite_task_name):
        """
        添加任务之间的依赖关系
        
        参数:
            task_name (str): 依赖任务的名称
            prerequisite_task_name (str): 前置任务的名称
            
        返回:
            bool: 成功返回True，任务不存在返回False
        """
        # 检查两个任务是否都存在
        if task_name not in self.dependencies or prerequisite_task_name not in self.dependencies:
            return False
        
        # 添加依赖关系
        self.dependencies[task_name].add(prerequisite_task_name)
        self.reverse_deps[prerequisite_task_name].add(task_name)
        return True
    
    def get_executable_tasks(self):
        """
        获取所有可执行的任务（没有依赖的任务）
        
        返回:
            list: 按优先级排序的可执行任务列表
        """
        executable = []
        for task_name, deps in self.dependencies.items():
            if not deps:  # 没有依赖
                # 在堆中找到任务
                for i in range(self.scheduler.size):
                    if self.scheduler.heap[i].name == task_name:
                        executable.append(self.scheduler.heap[i])
                        break
        
        # 按优先级排序
        executable.sort(key=lambda x: x.priority, reverse=True)
        return executable
    
    def execute_task(self, task_name):
        """
        执行特定任务（如果没有依赖）
        
        参数:
            task_name (str): 要执行的任务名称
            
        返回:
            bool: 成功返回True，如果任务不能执行则返回False
        """
        if task_name not in self.dependencies or self.dependencies[task_name]:
            return False  # 任务不存在或有依赖
        
        # 执行任务
        task = None
        for i in range(self.scheduler.size):
            if self.scheduler.heap[i].name == task_name:
                task = self.scheduler.heap[i]
                break
        
        if task is None:
            return False
        
        self.scheduler.delete_task(task_name)
        
        # 更新依赖这个任务的其他任务
        if task_name in self.reverse_deps:
            for dependent in self.reverse_deps[task_name]:
                if dependent in self.dependencies:
                    self.dependencies[dependent].remove(task_name)
            
            # 清理
            del self.reverse_deps[task_name]
        
        del self.dependencies[task_name]
        
        return True
    
    def execute_highest_priority_executable_task(self):
        """
        执行优先级最高且可执行的任务
        
        返回:
            MarketingTask: 执行的任务，如果没有可执行任务则返回None
        """
        executable = self.get_executable_tasks()
        if not executable:
            return None
        
        highest = executable[0]
        self.execute_task(highest.name)
        return highest
    
    def __str__(self):
        """依赖感知调度器的字符串表示"""
        result = ["任务依赖关系:"]
        
        for task_name, deps in self.dependencies.items():
            if deps:
                dep_str = ", ".join(deps)
                result.append(f"{task_name} 依赖于: {dep_str}")
            else:
                result.append(f"{task_name} (可执行)")
        
        result.append("\n任务优先级:")
        result.append(str(self.scheduler))
        
        return "\n".join(result)
