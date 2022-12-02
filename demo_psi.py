# -*- coding: utf-8 -*- 
# @Time : 2022/11/1 12:50 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com

import secretflow as sf
import numpy as np
from sklearn.datasets import load_iris
import os
import pandas as pd

# 数据质量最好通过每个源自行保证（预处理工程）
# 处理好的数据进行多源求交方便后续的操作
# In case you have a running secretflow runtime already.
# sf.shutdown()
os.chdir('/home/root-demo1/code/secretflow_demo')
# -------------------- 模拟提供的三个数据集 ---------------
os.makedirs('./data', exist_ok=True)
#
data, target = load_iris(return_X_y=True, as_frame=True)
data['uid'] = np.arange(len(data)).astype('str')
data['month'] = ['Jan'] * 75 + ['Feb'] * 75
da, db, dc = data.sample(frac=0.9), data.sample(frac=0.8), data.sample(frac=0.7)

# 生成数据保存->后续输入数据
da.to_csv('./data/alice.csv', index=False)
db.to_csv('./data/bob.csv', index=False)
dc.to_csv('./data/carol.csv', index=False)


# # ----------------- 虚拟化逻辑设备 ------------------
# sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=False)
# alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')
#
# # ----------------- 单多键隐私求交 -------------------
# input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv'}
# output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv'}
# spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
# # 合并方式为uid
# spu.psi_csv(key='uid', input_path=input_path, output_path=output_path, receiver='alice')
# # 合并方式为uid, month
# spu.psi_csv(key=['uid', 'month'], input_path=input_path, output_path=output_path, receiver='alice')
#
# # ------------- 三方隐私求交 ---------------
# input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv', carol: '.data/carol.csv'}
# output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv', carol: '.data/carol_psi.csv'}
#
# spu_3pc = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))
# spu_3pc.psi_csv(key=['uid', 'month'], input_path=input_path, output_path=output_path, receiver='alice', protocol='ECDH_PSI_3PC')