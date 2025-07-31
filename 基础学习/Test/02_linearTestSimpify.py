from mxnet import autograd, np, npx,gluon  
from d2l import mxnet as d2l
npx.set_np()
true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))  # 获取一个批次的数据
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1)) 
from mxnet import init
net.initialize(init.Normal(sigma=0.01))  # 初始化参数
loss = gluon.loss.L2Loss()  # 定义损失函数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})  # 定义优化器
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)  # 计算损失
        l.backward()  # 反向传播
        trainer.step(batch_size)  # 更新参数
    train_l = loss(net(features), labels)  # 计算训练集损失
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    # 将MXNet NDArray转换为NumPy数组后再计算误差
    print(f'error in estimating w: {net[0].weight.data()}')
    print(f'error in estimating b: {net[0].bias.data()}')