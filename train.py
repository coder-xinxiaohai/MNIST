import torch
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import Net
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

BATCH_SIZE = 50
EPOCH = 10

# 数据集预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data/', # root为数据集存放路径
    transform=transform,
    train=True,
    download=True
)
validate_dataset = datasets.MNIST(
    root='./data/',
    transform=transform,
    train=False,
    download=True
)

# 数据分批
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # 随机打乱训练集
    drop_last=False # 是否将最后一个不足batch的数据丢弃，默认为False。
)
validate_loader = Data.DataLoader(
    dataset=validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # 验证集的顺序不打乱
    drop_last=False # 是否将最后一个不足batch的数据丢弃，默认为False。
)

# # 查看训练集某一图片及其标签
# print(type(train_dataset.train_data[1])) # <class 'torch.Tensor'>
# # print(train_dataset.train_data[1]) # 手写字符集图片背景为黑色，前景为灰色。
# print(train_dataset.train_data[1].shape) # torch.Size([28, 28])
# print(type(train_dataset.train_labels[1])) # <class 'torch.Tensor'>
# print(train_dataset.train_labels[1]) # tensor(0)
# print(train_dataset.train_labels[1].shape) # torch.Size([]) 零维张量视为拥有张量属性的单独的一个数

# # 查看训练集的图片类型及尺寸
# print(type(train_dataset.train_data[1])) # <class 'torch.Tensor'>
# print(train_dataset.train_data[1].shape) # torch.Size([28, 28])

# # 绘制训练集中的一张图片
# # fig = plt.figure() # 创建画板
# plt.imshow(train_dataset.train_data[1])
# # plt.imshow(train_dataset.train_data[1], cmap='gray')
# plt.xticks([]) # 不显示横坐标
# plt.yticks([]) # 不显示纵坐标
# # plt.savefig('./data/pic.jpg')
# plt.show()

# # 展示MNIST数据集的前12幅图片
# # fig = plt.figure() # 创建画板
# for i in range(12):
#     plt.subplot(3, 4, i+1) # 第一个参数表示行数；第二个参数表示列数；第三个参数表示第几个图。
#     plt.tight_layout() # 该函数要在所有画图函数之前，在plt.show()之后。用于解决子图重叠问题。
#     plt.imshow(train_dataset.train_data[i])
#     # plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels:{}".format(train_dataset.train_labels[i]))
#     plt.xticks([]) # 不显示横坐标
#     plt.yticks([]) # 不显示纵坐标
# plt.show()


learning_rate = 0.01
momentum = 0.5

# 创建一个Net网络对象net
net = Net()
criterion = nn.CrossEntropyLoss() # 交叉熵损失

# 在反向传播计算完所有参数的梯度后，还需要使用优化方法更新网络的权重和参数。
# torch.optim中实现了深度学习中绝大多数的优化方法，例如SGD。
# 新建一个优化器，设置学习率，并指定要调整的参数。
# PyTorch将深度学习中常用的优化方法全部封装在torch.optim中，其设计十分灵活，能够很方便地扩展成自定义的优化方法。
# 所有的优化方法都是继承基类optim.Optimizer，并实现了自己的优化步骤。
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum) # 随机梯度下降
save_path = './Net.pth' # 保存模型

val_num = 10000
best_acc = 0.0
# 迭代10次
for epoch in range(EPOCH):
    net.train() # 训练阶段用net.train()
    running_loss = 0.0 # 计算训练阶段一个batch的平均损失
    t1 = time.perf_counter() # 返回当前计算机系统的时间
    # 循环训练集，从1开始。
    for step, data in enumerate(train_loader, start = 1):
        inputs, labels = data # data是一个列表，[数据，标签]
        # inputs.shape: torch.Size([50, 1, 28, 28])
        # labels.shape: torch.Size([50])
        optimizer.zero_grad() # 优化器的梯度清零，每次循环都要清零。
        outputs = net(inputs) # 输出为标签的预测值
        loss = criterion(outputs, labels)
        loss.backward() # loss进行反向传播
        optimizer.step() # step进行参数更新
        running_loss += loss.item() # item()返回loss的值，每次计算完loss后加入到running_loss中，可以算出叠加之后的总loss
    print(time.perf_counter()-t1) # 记录训练一个epoch所需要的时间

    net.eval() # 非训练阶段用net.eval()
    acc = 0.0
    with torch.no_grad():
    # 在使用PyTorch时，并不是所有的操作都需要进行计算图的生成(计算过程的构建，以便梯度反向传播等操作)
    # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with.torch.no_grad():
    # 强制之后的内容不进行计算图的构建
        for data_validate in validate_loader:
            images, labels = data_validate
            outputs = net(images)
            predict_y = torch.max(outputs, dim=1)[1] # dim=1表示行 dim=0表示列 网络预测为对应类别的概率 而行代表样本 列代表类别
            # 输出每行最大值所对应的索引，即计算模型中每个类别的最大值并返回其索引值，即该类别的标签值
            acc += (predict_y == labels).sum().item() # .sum就是将所有值相加，得到的仍然是tensor，使用.item()之后得到的就是值。
        acc_validate = acc/val_num
        if acc_validate > best_acc:
            best_acc = acc_validate
            torch.save(net.state_dict(), save_path) # 只保存网络中的参数

            # PyTorch保存模型与加载
            # 模型的保存
            # torch.save(net,PATH) # 保存模型的整个网络，包括网络的整个结构和参数
            # torch.save(net.state_dict, PATH) # 只保持网络中的参数

            # 模型的加载
            # 分别对应上边的加载方法
            # model_dict=torch.load(PATH)
            # model_dict=net.load.dict(torch.load(PATH)

        print('[epoch %d] train_loss: %.3f validate_accuracy: %.3f' % (epoch + 1, running_loss / step, acc_validate))
print('Finish Training')

