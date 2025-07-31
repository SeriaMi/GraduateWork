import sys
from mxnet import gluon,autograd
from d2l import mxnet as d2l
import numpy as np
from IPython import display
mnist_train = gluon.data.vision.datasets.FashionMNIST(train=True)
mnist_test = gluon.data.vision.datasets.FashionMNIST(train=False)
print(mnist_train[0][0].shape)
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i]) 
    return axes
X, y = mnist_train[:18]
# print(X.shape)
show_images(X.squeeze(axis = -1),2,9,titles=get_fashion_mnist_labels(y))

# 添加以下显示命令
import matplotlib.pyplot as plt
# plt.show()

batch_size = 256
def get_dataloader_workers():
    return 0 #使用4个进程来读取数据
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer), 
                                    batch_size, shuffle=True, 
                                    num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'Time cost: {timer.stop():.2f} sec')
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the fashion mnist dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                 num_workers=get_dataloader_workers()))

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
# W.attach_grad()
# b.attach_grad()
def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition
X = np.random.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))
y = np.array([0, 2])
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
def net(X):

    # 将输入展平为二维：batch_size × 784
    # 原代码中的reshape产生了多余的维度，改为显式指定展平后的形状
    return softmax(np.dot(X.reshape((X.shape[0], -1)), W) + b)
def cross_entropy(y_hat, y):
    return -np.log(y_hat[range(len(y_hat)), y])
#y表示每个样本的标签，比如t_shirt对应0，trouser对应2,y_hat表示预测的概率，
# 每一行y_hat[i][d]表示第i个样本各个类别(d)的概率
def accuracy(y_hat, y):
    """计算预测正确的样本数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #y_hat.argmax(axis=1) 的作用是获取二维数组y中每一行的最大值所在的列索引
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())
#Test
# y = np.array([2, 0, 4], dtype=np.int32)
# y_hat = np.array([2, 1, 4], dtype=np.int64)
# # 转换后比较
# cmp = y_hat.astype(np.int32) == y
# # 结果：array([ True, False,  True])
# print(cmp.astype(np.int32).sum()/len(cmp))
def evaluate_accuracy(data_iter, net):
    """计算在指定数据集上的分类准确率
    Args:
        data_iter: 数据迭代器（测试集或验证集）
        net: 神经网络模型
    Returns:
        分类准确率（正确样本数/总样本数）
    """
    metric = Accumulator(2)  # 创建累加器，跟踪[正确数, 总数]
    for X, y in data_iter:   # 遍历数据集
        metric.add(accuracy(net(X), y), y.size)  # 累计：当前批正确数 + 当前批样本数
    return metric[0] / metric[1]  # 总正确数 / 总样本数

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

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    metric = Accumulator(3)  # 训练损失、训练准确率、样本数
    if isinstance(updater, gluon.Trainer):
        updater = updater.step # “忽略”batch_size参数
    for X, y in train_iter:
        # 计算梯度并更新参数
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图中添加多个数据点"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(test_iter, net)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
