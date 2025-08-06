import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.randn(2, 20)
# print(net(x))
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.hidden = nn.Linear(20, 256) 
#         self.out = nn.Linear(256, 10)
#     def forward(self, x):
#         return self.out(F.relu(self.hidden(x)))
# net = MLP()
# print(net(x))
#访问参数

# def block0():
#     return nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# def block1():
#     net = nn.Sequential()
#     for i in range(3):
#         net.add_module(f'block{i}', block0())
#     return net
# rgnet = nn.Sequential(block1(), nn.Linear(10, 2))
# print(rgnet)
# def init_normal(m):
#     # 参数初始化函数，对神经网络模块进行初始化操作
#     if type(m) == nn.Linear:  # 修正：type(m) 判断模块类型是否为全连接层
#         nn.init.normal_(m.weight, mean=0, std=0.01)  # 使用正态分布初始化权重（均值0，标准差0.01）
#         if m.bias is not None:  # 如果存在偏置项
#             nn.init.zeros_(m.bias)  # 将偏置项初始化为全0
# net.apply(init_normal)  # 应用初始化函数到网络的所有子模块（递归遍历所有层）
# shared = nn.Linear(20, 20)  # 定义一个共享的线性层
# net1 = nn.Sequential(shared, shared, nn.Linear(10, 2))  # 定义网络1
# print(net1[0].weight.data()== net1[1].weight.data())
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# x = torch.ones(2,3,device =try_gpu())
# print(x.device)
# print(x)
# Y = torch.rand( 2, 3, device=try_gpu())

# #计算X+Y的和
# Z = x + Y
# print(Z)
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net.to(try_gpu())
x = torch.randn(2, 20, device=try_gpu())
print(net(x))