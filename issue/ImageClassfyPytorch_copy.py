import secretflow as sf
import spu
from secretflow.data.ndarray import load
from secretflow.security import SPUAggregator
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

# ray start --head --node-ip-address=xx --port=6379 --resources='{"alice": 2,"bob":2}' --include-dashboard=True
# ray start --address='xx:6379'  --resources='{"bob": 2}'

# ray start --head --node-ip-address="xx" --port="6379" --resources='{"alice": 16}' --include-dashboard=False --disable-usage-stats --num-cpus=16
# ray start --address="xx:6379" --resources='{"bob": 16}' --disable-usage-stats

# sf.shutdown()
#
# sf.init(address="xx:6379")
#
# cluster_def = {
#         'nodes': [
#             {
#                 'party': 'alice',
#                 'id': '0',
#                 'address': 'xx:8285',
#             },
#             {
#                 'party': 'bob',
#                 'id': '1',
#                 'address': 'xx:8285',
#             },
#         ],
#         'runtime_config': {
#             'protocol': spu.spu_pb2.SEMI2K,
#             'field': spu.spu_pb2.FM128,
#             'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
#         }
#     }
#
# spu = sf.SPU(cluster_def=cluster_def)
# alice=sf.PYU("alice")
# bob=sf.PYU("bob")


sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], num_cpus=8, log_to_driver=False)
alice, bob = sf.PYU('alice'), sf.PYU('bob')

# train_data = load({bob: 'x_train1.npy', alice: 'x_train2.npy'})
# test_data = load({bob: 'x_test1.npy', alice: 'x_test2.npy'})
# train_label = load({bob: 'y_train1.npy', alice: 'y_train2.npy'})
# test_label = load({bob: 'y_test1.npy', alice: 'y_test2.npy'})
print("data load successfully!")

def load_dataset(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    # normalized_x: bool = True,
    # categorical_y: bool = False,
    is_torch: bool = False,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:


    x_train, y_train = np.load("x_train1.npy"), np.load("y_train1.npy")
    x_test, y_test = np.load("x_test1.npy"), np.load("y_test1.npy")

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

(train_data, train_label), (test_data, test_label) = load_dataset(
    parts={alice: 0.4, bob: 0.6},
    # normalized_x=True,
    # categorical_y=True,
    is_torch=True,
)


# import os
# os.environ['CUDA_VISIBLE_DEVICE']='1'
# import torch
# torch.cuda.set_device(1)

from torch import nn, optim
from secretflow.ml.nn.fl.backend.torch.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.security.aggregation import SecureAggregator
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision


class ConvNet(BaseModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(16 * 28 * 28, 5),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255
        x = self.model(x)
        return x


loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, num_classes=5, average='micro'),
        metric_wrapper(Precision, num_classes=5, average='micro'),
    ],
)
print("model create")

device_list = [alice, bob]
server = alice
aggregator = SecureAggregator(server, [alice, bob])
print("聚合success")

# spcify params
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy='fed_avg_w',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
)
print("FLModel load success")

history = fl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=20,
    batch_size=8,
    aggregate_freq=1,
)
print("train finished")

fl_model.save_model({alice: "flower_classification.pth"})

import matplotlib

matplotlib.use('AGG')  # 或者PDF, SVG或PS
from matplotlib import pyplot as plt

# Draw accuracy values for training & validation
plt.plot(history.global_history['accuracy'])
plt.plot(history.global_history['val_accuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("accuracy.png")

plt.plot(history.global_history['loss'])
plt.plot(history.global_history['val_loss'])
plt.title('FLModel loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('loss.png')
