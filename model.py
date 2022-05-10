import torch
import torch.nn as nn

# torch.nn是专门为神经网络设计的模块化接口。nn.Module是nn中最重要的类，可以把它看作一个网络的封装，
# 包含网络各层定义及forward方法，调用forward(input)方法，可以返回前向传播的结果。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # input:[1, 28, 28]    output:[10, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # input:[10, 24, 24]    output:[10, 12, 12]
        )

        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), # input:[10, 12, 12]    output:[20, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # input:[20, 8, 8]    output:[20, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 10) # 手写字符集识别问题为10分类问题。
        )

    # 只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现。
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1) # 展平处理(除去batch维度)
        x = self.fc(x)
        return x

# 在使用PyTorch封装好的网络层时，不需要对模型的参数初始化，因为这些PyTorch都会帮助我们完成。
# 但是如果是我们自己搭建模型，不使用PyTorch中的封装好的网络层或者对PyTorch中封装好的模型初始化参数不满意，此时我们就需要对模型进行参数初始化。