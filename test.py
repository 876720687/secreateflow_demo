#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/12/20 17:20
import transform as transform
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
# from torch_geometric.loader import DataLoader
from torchvision.datasets import CIFAR10


import hashlib
import os
import pickle
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy

from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.utils.hash import sha256sum
from secretflow.utils.simulation.data.dataframe import create_df, create_vdf
from secretflow.utils.simulation.data.ndarray import create_ndarray


#
# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize([0.5], [0.5])])
#
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=data_tf, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
#




# trainloader = t.utils.data.DataLoader(
#     trainset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=0
# )
#
# testloader = t.utils.data.DataLoader(
#     testset,
#     batch_size=4,
#     shuffle=False,
#     num_workers=0
# )



def load_fashion_mnist(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
    # normalized_x: bool = True,
    # categorical_y: bool = False,
    is_torch: bool = False,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:
    """Load mnist dataset to federated ndarrays.

    This dataset has a training set of 60,000 examples, and a test set of 10,000 examples.
    Each example is a 28x28 grayscale image of the 10 digits.
    For the original dataset please refer to `MNIST <http://yann.lecun.com/exdb/mnist/>`_.

    Args:
        parts: the data partitions. The dataset will be distributed as evenly
            as possible to each PYU if parts is a array of PYUs. If parts is a
            dict {PYU: value}, the value shall be one of the followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the right-side.
        normalized_x: optional, normalize x if True. Default to True.
        categorical_y: optional, do one hot encoding to y if True. Default to True.

    Returns:
        A tuple consists of two tuples, (x_train, y_train) and (x_train, y_train).
    """
    # filepath = _get_dataset(_DATASETS['mnist'])
    # with np.load(filepath) as f:
    #     x_train, y_train = f['x_train'], f['y_train']
    #     x_test, y_test = f['x_test'], f['y_test']
    #
    # if normalized_x:
    #     x_train, x_test = x_train / 255, x_test / 255
    #
    # if categorical_y:
    #     from sklearn.preprocessing import OneHotEncoder
    #
    #     encoder = OneHotEncoder(sparse=False)
    #     y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    #     y_test = encoder.fit_transform(y_test.reshape(-1, 1))

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

    # 测试集
    testset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    x_train, y_train = trainset.data, trainset.target
    x_test, y_test = testset.data, testset.target

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