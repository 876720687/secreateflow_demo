# -*- coding: utf-8 -*- 
# @Time : 2022/11/1 12:50 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com

import warnings
warnings.filterwarnings("ignore")

import secretflow as sf
import numpy as np
from sklearn.datasets import load_iris
import os
import pandas as pd


# ----------------- 虚拟化逻辑设备 ------------------
# sf.init(['alice', 'carol'], num_cpus=8, log_to_driver=False)
# sf.init(address='auto')
alice, bob, carol = sf.PYU('alice'), sf.PYU('carol')
'''
import spu
import secretflow as sf
# Use ray head adress
sf.init(address='192.168.200.203:9394')

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            'id': '0',
            # Use the address and port of alice instead.
            # Please choose a unused port.
            'address': '192.168.200.203:4040',
        },
        {
            'party': 'carol',
            'id': '2',
            # Use the ip and port of bob instead.
            # Please choose a unused port.
            'address': '192.168.200.205:4041',
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}

spu = sf.SPU(cluster_def=cluster_def)
'''
aby2_config = sf.utils.testing.cluster_def(parties=['alice', 'carol'])
spu = sf.SPU(aby2_config)

# alice, carol = sf.PYU('alice'), sf.PYU('carol')
# ----------------- 单多键隐私求交 -------------------
input_path = {alice: '.data/alice.csv', carol: '/home/almalinux/sf-benchmark/carol.csv'}
output_path = {alice: '.data/alice_psi.csv', carol: '.data/carol_psi.csv'}
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'carol']))
# 合并方式为uid
spu.psi_csv(key='uid', input_path=input_path, output_path=output_path, receiver='alice')
# 合并方式为uid, month
spu.psi_csv(key=['uid', 'month'], input_path=input_path, output_path=output_path, receiver='alice')
