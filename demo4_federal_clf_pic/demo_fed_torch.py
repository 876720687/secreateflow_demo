# -*- coding: utf-8 -*-
# @Time : 2022/12/5 16:32 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com

# TODO： needs high performance server.
# success in server.

import secretflow as sf
from matplotlib import pyplot as plt

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], num_cpus=8, log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

from secretflow.ml.nn.fl.backend.torch.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
from torch.nn import functional as F


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return F.softmax(x, dim=1)


(train_data, train_label), (test_data, test_label) = load_mnist(
    parts={alice: 0.4, bob: 0.6},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)

loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
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
    strategy='fed_avg_w', # fl strategy
    backend="torch", # backend support ['tensorflow', 'torch']
)

history = fl_model.fit(
            train_data,
            train_label,
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

