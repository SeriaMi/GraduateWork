import torch
x = torch.arange(4.0)
print(x)
x.requires_grad_(True) #之后即可访问x.grad属性
# 等价于x = torch.arange(4.0, requires_grad=True)
# y = 2 * torch.dot(x, x)
# print(y)
# y.backward() #计算y对x的梯度,即求导
# print(x.grad) #dy/da
# x.grad.zero_() #清除梯度
# y = x.sum()
# y.backward() #计算y对x的梯度
# print(x.grad) #dy/da
# y = x * x
# print(y)
# y.mean().backward()
# print(x.grad) #dy/da
# x.grad.zero_()
# y = x*x
# u = y.detach()
# print(u)
# Z = u*x
# Z.sum().backward()
# print(x.grad == u)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)#size=()表示标量
d = f(a)
d.backward()
print(a.grad == d / a) #验证梯度是否正确 