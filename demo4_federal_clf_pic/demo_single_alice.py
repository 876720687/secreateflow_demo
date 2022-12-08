# -*- coding: utf-8 -*- 
# @Time : 2022/11/3 10:35 
# @Author : YeMeng 
# @File : demo_single_model.py 
# @contact: 876720687@qq.com
import numpy as np
from secretflow.ml.nn.fl.backend.torch.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import optim_wrapper, metric_wrapper
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from secretflow.utils.simulation.datasets import load_mnist, dataset
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, Precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


mnist = np.load(dataset('mnist'), allow_pickle=True)
image = mnist['x_train']
label = mnist['y_train']

alice_x = image[:10000]
alice_y = label[:10000]
alice_y = OneHotEncoder(sparse=False).fit_transform(alice_y.reshape(-1, 1))




# def create_model(input_shape,num_classes):
#
#     model = keras.Sequential(
#         [
#             keras.Input(shape=input_shape),
#             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation="softmax"),
#         ]
#     )
#     # Compile model
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=["accuracy"])
#     return model
#
#
# num_classes = 10
# input_shape = (28, 28, 1)
# single_model = create_model(input_shape=input_shape,num_classes=num_classes)



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

# Compile model
model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=CrossEntropyLoss,
    optim_fn=optim_wrapper(optim.Adam, lr=5e-3),
    metrics=[
        metric_wrapper(Accuracy, num_classes=3, average='micro'),
        metric_wrapper(Precision, num_classes=3, average='micro'),
    ],
)






alice_X_train, alice_X_test, alice_y_train, alice_y_test = train_test_split(alice_x,
    alice_y, test_size=0.33, random_state=1234)

single_model.fit(alice_X_train, alice_y_train, validation_data=(alice_X_test, alice_y_test), batch_size=128, epochs=10)


