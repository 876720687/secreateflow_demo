# -*- coding: utf-8 -*-
# @Time : 2022/12/5 16:32 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com

# TODO： needs high performance server.
# success in server.

import secretflow as sf
from secretflow.ml.nn.fl.backend.torch.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
from torch.nn import functional as F
from secretflow.utils.simulation.data.ndarray import create_ndarray
import numpy as np
import torchvision.transforms as transforms
# from torch_geometric.loader import DataLoader
from torchvision.datasets import CIFAR10
from typing import Dict, List, Tuple, Union
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.utils.hash import sha256sum
from secretflow.utils.simulation.data.dataframe import create_df, create_vdf
from secretflow.utils.simulation.data.ndarray import create_ndarray


# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], num_cpus=8, log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


# class ConvNet(BaseModule):
#     """Small ConvNet for MNIST."""
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
#         self.fc_in_dim = 192
#         self.fc = nn.Linear(self.fc_in_dim, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 3))
#         x = x.view(-1, self.fc_in_dim)
#         x = self.fc(x)
#         return F.softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # nn. Module. init (self)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
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

# class Batch_Net(nn.Module):
#     """
#     在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
#     """
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(Batch_Net, self).__init__()
#         self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
#         self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
#         self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


def load_fashion_mnist(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    # normalized_x: bool = True,
    # categorical_y: bool = False,
    is_torch: bool = False,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 训练集
    trainset = CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    # 测试集
    testset = CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    x_train, y_train = trainset.data, np.array(trainset.targets)
    x_test, y_test = testset.data, np.array(testset.targets)
    # arr = arr[:, np.newaxis, :, :]
    # ndarray->Fedndarray. but why torch?
    return (
        (
            create_ndarray(x_train, parts=parts, axis=0, is_torch=is_torch),
            create_ndarray(y_train, parts=parts, axis=0),
        ),
        (
            create_ndarray(x_test, parts=parts, axis=0, is_torch=is_torch),
            create_ndarray(y_test, parts=parts, axis=0),
        ),
    )

# (train_data, train_label), (test_data, test_label) = load_mnist(
#     parts={alice: 0.4, bob: 0.6},
#     normalized_x=True,
#     categorical_y=True,
#     is_torch=True,
# )

(train_data, train_label), (test_data, test_label) = load_fashion_mnist(
    parts={alice: 0.4, bob: 0.6},
    # normalized_x=True,
    # categorical_y=True,
    is_torch=True,
)


model_def = TorchModel(
    model_fn=Net,
    loss_fn=nn.CrossEntropyLoss,
    optim_fn=optim_wrapper(optim.Adam, lr=1e-2),
    metrics=[
        metric_wrapper(Accuracy, num_classes=10, average='micro'),
        metric_wrapper(Precision, num_classes=10, average='micro'),
    ],
)

device_list = [alice, bob]
server = charlie
aggregator = SecureAggregator(server,[alice,bob])

# spcify params
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy='fed_avg_w',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
)

history = fl_model.fit(
            x=train_data,
            y=train_label,
            validation_data=(test_data, test_label),
            epochs=20,
            batch_size=32,
            aggregate_freq=1,
        )



# # Draw accuracy values for training & validation
# plt.plot(history.global_history['accuracy'])
# plt.plot(history.global_history['val_accuracy'])
# plt.title('FLModel accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'], loc='upper left')
# plt.show()

