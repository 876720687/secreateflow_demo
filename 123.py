#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/20 19:43
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
# from torch_geometric.loader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转换为Image,方便可视化
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 训练集
trainset = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)
# 测试集
testset = CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# datasets对象是一个数据集,可以按下标访问,返回形如(data,labe)的数据
(data, label) = trainset[100]
data.size()  # 验证某一张图片的维度
print(classes[label])
show((data + 1) / 2).resize((100, 100))
# Dataloader是一个可迭代的对象,它将 datasets返回的每一条数据拼接成一个 batch,并提供
# 多线程加速优化和数据打乱等操作。当程序将 datasets中的所有数据遍历完一遍之后,即对 Dataloader完成了一次迭代
dataiter = iter(trainloader)
images, labels = dataiter.next()  # 返回4张图片及标签
print("-----------------")
print(''.join('%11s' % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100))


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
imshow(tv.utils.make_grid(images))
plt.show()


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # nn. Module. init (self)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fcl = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fcl(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# 定义损失函数和优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
import time

start_time = time.time()
for epoch in range(2):
    running_loss = 0.0
    # img, label = data
    # #img.size(0)表示每个batch图片数量
    # img = img.view(img.size(0), -1)
    # #img.view():与reshape类似，用来转换size大小,-1表示会自适应的调整剩余的维
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d]loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
end_time = time.time()
print("Training duration:", end_time - start_time)
dataiter = iter(testloader)
images, labels = dataiter.next()
print('True label:', ''.join('%08s' % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))
outputs = net(images)
_, predicted = t.max(outputs.data, 1)
print('Predicted label:', ''.join('%06s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('Accuracy in testset of 10000 images:%d %%' % (100 * correct / total))