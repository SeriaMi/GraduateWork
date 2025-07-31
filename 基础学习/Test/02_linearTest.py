import random
from mxnet import autograd, np,npx
from d2l import mxnet as d2l
def synthetic_data(w, b, num_examples):
    """Generate synthetic data y = Xw + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)  # Add some noise
    return X, y.reshape(-1, 1)
true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
# d2l.plt.show()  # 添加这行来显示图形

#read dataset
def data_iter(batch_size, features, labels):
    """生成小批量数据样本
    参数：
        batch_size: 每批样本数量
        features: 特征矩阵 (num_examples, num_features)
        labels: 标签向量 (num_examples, 1)
    返回：
        生成器对象，每次产生 (features_batch, labels_batch)
    """
    num_examples = len(features)          # 获取样本总数
    indices = list(range(num_examples))   # 生成样本索引列表[0,1,...,n-1]
    random.shuffle(indices)               # 随机打乱索引顺序
    print(len(features))
    # 按批次遍历所有样本
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引切片
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        # 生成当前批次的特征和标签 (自动转换为NDArray)
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    # print( X,'\n' ,y)
    pass
# Initialize model parameters
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()  # Enable gradient tracking
b.attach_grad()  # Enable gradient tracking
def linreg(X, w, b):
    """线性回归模型"""
    return np.dot(X, w) + b
def squared_loss(y_hat, y):
    """平方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    参数：
        params: 模型参数列表 [w, b, ...]
        lr: 学习率 (learning rate)
        batch_size: 当前批次样本数量
    实现：
        使用 param[:] 进行原地更新，保持NDArray数据类型不变
        梯度需除以batch_size，因损失计算的是批次样本的平均损失
    """
    for param in params:
        # 参数更新公式：param = param - lr * (梯度平均值)
        param[:] = param - lr * param.grad / batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            # 前向计算
            l = loss(net(X, w, b), y)
        # 反向传播
        l.backward()
        # 更新参数
        sgd([w, b], lr, batch_size)
    # 计算当前epoch的损失
    train_l = loss(net(features, w, b), labels).mean()
    print(f'epoch {epoch + 1}, loss {float(train_l):f}')
    print(f'weight {w.T}, bias {b}')