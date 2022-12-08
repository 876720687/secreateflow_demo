# -*- coding: utf-8 -*- 
# @Time : 2022/11/3 10:18 
# @Author : YeMeng 
# @File : demo.py 
# @contact: 876720687@qq.com
import numpy as np
import secretflow as sf
from matplotlib import pyplot as plt
from secretflow.ml import nn
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.fl.backend.torch.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import optim_wrapper, metric_wrapper
from secretflow.security import SecureAggregator, SPUAggregator
from secretflow.utils.simulation.datasets import load_mnist, dataset
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(parties=['alice', 'bob', 'charlie'], num_cpus=4, log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')
# alice和bob角色是client，charlie角色是server
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# ----------- 构建数据集 ---------------
(x_train, y_train), (x_test, y_test) = load_mnist(
    parts=[alice, bob],
    normalized_x=True,
    categorical_y=True)

mnist = np.load(dataset('mnist'), allow_pickle=True)
image = mnist['x_train']
label = mnist['y_train']

# ----------------- model def -----------------

def create_conv_model(input_shape, num_classes, name='model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        # Create model
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model
num_classes = 10
input_shape = (28, 28, 1)
model = create_conv_model(input_shape, num_classes)




# def create_conv_model():
#     def create_model():
#         class ConvNet(BaseModule):
#             """Small ConvNet for MNIST."""
#             def __init__(self):
#                 super(ConvNet, self).__init__()
#                 self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
#                 self.fc_in_dim = 192
#                 self.fc = nn.Linear(self.fc_in_dim, 10)
#
#             def forward(self, x):
#                 x = F.relu(F.max_pool2d(self.conv1(x), 3))
#                 x = x.view(-1, self.fc_in_dim)
#                 x = self.fc(x)
#                 return F.softmax(x, dim=1)
#
#         # Compile model
#         model_def = TorchModel(
#             model_fn=ConvNet,
#             loss_fn=nn.CrossEntropyLoss,
#             optim_fn=optim_wrapper(optim.Adam, lr=5e-3),
#             metrics=[
#                 metric_wrapper(Accuracy, num_classes=3, average='micro'),
#                 metric_wrapper(Precision, num_classes=3, average='micro'),
#             ],
#         )
#         return model
#
#     return create_model
#
# model = create_conv_model()





# -------------- 聚合 --------------
device_list = [alice, bob]
secure_aggregator = SecureAggregator(charlie, [alice, bob])
spu_aggregator = SPUAggregator(spu)
fed_model = FLModel(server=charlie,
                    device_list=device_list,
                    model=model,
                    aggregator=secure_aggregator,
                    strategy="fed_avg_w",
                    backend="tensorflow")

history = fed_model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        sampler_method="batch",
                        batch_size=128,
                        aggregate_freq=1)





## --------------- Draw accuracy values for training & validation
plt.plot(history.global_history['accuracy'])
plt.plot(history.global_history['val_accuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# Draw loss for training & validation
plt.plot(history.global_history['loss'])
plt.plot(history.global_history['val_loss'])
plt.title('FLModel loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

global_metric = fed_model.evaluate(x_test, y_test, batch_size=128)
print(global_metric)