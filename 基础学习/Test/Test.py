import numpy as np
import math
import matplotlib.pyplot as plt

def seekmax(x):
    temp = x[0]
    for i in range(len(x)):
        if x[i] > temp:
            temp = x[i]
    return temp


y = np.random.normal(0,1,(3,5))
print(y)
print(seekmax(y[0]))
print(seekmax(y[1]))
print(seekmax(y[2]))
y = y.argmax(axis=1)
print(y)
x = np.array([1, 2, 3, 4, 5])
class Accumulator:
    """在n个变量上累加的实用工具类
    用于跟踪训练过程中的多个指标（如损失、准确率等）"""
    
    def __init__(self, n):
        self.data = [0.0] * n  # 初始化存储空间（浮点型列表）
        
    def add(self, *args):
        # 将输入参数逐个累加到对应位置（参数需与data长度匹配）
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)  # 重置所有累加值为0
        
    def __getitem__(self, idx):
        return self.data[idx]
    
metric = Accumulator(3)
metric.add(1, 2)
print(metric[0])
print(metric[1])
