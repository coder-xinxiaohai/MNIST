import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Net
import numpy as np

# 数据集预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

image = Image.open('./data/0.jpg')
# print(image.mode) # 查看输入图片模式
plt.imshow(image)
plt.show()

image = transform(image) # 数据预处理
# print(image.shape) # torch.Size([1, 28, 28])

# #展示经过图像预处理后的图片
# image = torch.squeeze(image, dim=0) # 去掉通道数
# # print(type(image)) # <class 'torch.Tensor'>
# # print(image.shape) # torch.Size([28, 28])
# plt.imshow(image)
# # plt.imshow(image, cmap='gray')
# plt.xticks([]) # 不显示横坐标
# plt.yticks([]) # 不显示纵坐标
# plt.show()

image = torch.unsqueeze(image, dim=0) # 增加了一个batch维度
# print(image.shape) # torch.Size([1, 1, 28, 28])

model = Net()
model_weight_path = 'Net.pth'
model.load_state_dict(torch.load(model_weight_path))

model.eval()
with torch.no_grad():
    # print(model(image))
    # print('model(image).shape:',model(image).shape) # model(image).shape: torch.Size([1, 10])
    output = torch.squeeze(model(image)) # 压缩batch维度
    # print('output.shape:',output.shape) # output.shape: torch.Size([10])
    predict = torch.softmax(output, dim=0) # 行归一化(即归一化的维度为行) softmax后就是一个概率分布
    # print('predict:',predict) # predict: tensor([7.9245e-01, 4.2556e-05, 3.3293e-03, 3.1710e-03, 7.0948e-05, 1.8576e-01, 8.0935e-03, 8.3387e-04, 2.8358e-04, 5.9647e-03])
    predict_class = torch.argmax(predict).numpy() # 返回概率最大值所对应的索引值
print('The number is:', predict_class,'!')
plt.show()