# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 19:38 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def breast_cancer(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    scaler = Normalizer(norm='max')
    x, y = load_breast_cancer(return_X_y=True)
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        if party_id:
            if party_id == 1:
                return x_train[:, 15:], 0
            else:
                return x_train[:, :15], y_train
        else:
            return x_train, y_train
    else:
        return x_test, y_test


