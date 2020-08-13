import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
from torch.autograd import Variable
import matplotlib.pylab as plt

torch.manual_seed(1)  # reproducible

# Hyper ParametersEPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 32  # rnn 时间步数 / 图片高度
INPUT_SIZE = 32 * 3  # rnn 每步输入值 / 图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = torchvision.datasets.CIFAR10(
    root='./data/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.CIFAR10(root='./data/', download=DOWNLOAD_MNIST, train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=INPUT_SIZE,  # 图片每行的数据像素点
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        print(r_out.shape)
        print(r_out[:, -1, :].shape)
        quit()
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
rnn.to('cuda')
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_list = []
# training and testing
for epoch in range(1):
    for step, (x, b_y) in enumerate(train_loader):  # gives batch data
        x = x.view(-1, 32, 32 * 3)  # reshape x to (batch, time_step, input_size)
        x = x.to('cuda')
        b_y = b_y.to('cuda')
        output = rnn(x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_list.append(loss.item())
        print(loss.item())

print("ok!")
x = range(len(loss_list))
plt.plot(x, loss_list)
plt.show()


# import torch
# import torchvision
# from torchvision import transforms
# from torch import nn
# from torch import optim
# from torch.autograd import Variable
# import torch.nn.functional as F
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
#
# trainsets = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainsets, batch_size=100, shuffle=True)
# testsets = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(trainsets, batch_size=100, shuffle=False)
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.LSTM = nn.LSTM(32 * 3, 128, batch_first=True, num_layers=3)  # 将彩色图片输入给LSTM怎么办
#         self.output = nn.Linear(128, 10)
#
#     def forward(self, x):
#         out, (h_n, c_n) = self.LSTM(x)
#         return self.output(out[:, -1, :])
#
#
# if __name__ == '__main__':
#     net = Net()
#     Loss = nn.CrossEntropyLoss()
#     Opt = optim.Adam(net.parameters(), lr=0.01)
#     for i in range(100):
#         for data, lable in trainloader:
#             data = Variable(data)
#             lable = Variable(lable)
#             data = data.view(-1, 32, 32 * 3)
#             out = net(data)
#             loss = Loss(out, lable)
#             Opt.zero_grad()
#             loss.backward()
#             Opt.step()
#             print(loss.data.item())
