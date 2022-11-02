# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 19:38 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
from typing import Sequence
import flax.linen as nn


FEATURES = [30, 15, 8, 1]


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

