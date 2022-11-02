# -*- coding: utf-8 -*- 
# @Time : 2022/11/1 12:50 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
import secretflow as sf
import numpy as np
from sklearn.datasets import load_iris
import os




# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=False)
data, target = load_iris(return_X_y=True, as_frame=True)
data['uid'] = np.arange(len(data)).astype('str')
data['month'] = ['Jan'] * 75 + ['Feb'] * 75

# -------------------- 模拟提供的三个数据集 ---------------
os.makedirs('.data', exist_ok=True)
da, db, dc = data.sample(frac=0.9), data.sample(frac=0.8), data.sample(frac=0.7)

da.to_csv('.data/alice.csv', index=False)
db.to_csv('.data/bob.csv', index=False)
dc.to_csv('.data/carol.csv', index=False)

# ----------------- 单隐私求交 ------------------
# 虚拟化三个逻辑设备
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv'}
output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv'}
spu.psi_csv('uid', input_path, output_path, 'alice')

# ----------------- 单隐私求交结果验证 ------------------
import pandas as pd

df = da.join(db.set_index('uid'), on='uid', how='inner', rsuffix='_bob', sort=True)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)


# -------------- 多键隐私求交 ----------------


# ------------- 三方隐私求交 ---------------