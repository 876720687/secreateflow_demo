# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 18:41 
# @Author : YeMeng 
# @File : demo_spu.py 
# @contact: 876720687@qq.com

# operation info import
import os
from pathlib import Path
# 添加路径的两种方法，后期更加推荐第二种
import sys
# sys.path.append("/home/almalinux/sf-benchmark/demo2_spu_logistic")
FILE = Path(__file__).resolve() # '/home/almalinux/sf-benchmark/demo2_spu_logistic/demo_spu.py'
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from demo_processJax import *
from config import *
import secretflow as sf


# ---------------- Spu ------------------
# sf.shutdown() # 防止集群是启动状态的
# 伪分布式
# sf.init(['alice', 'bob'], num_cpus=8,  log_to_driver=True)

# 分布式,先确定集群是启动的ray status
# sf.init(address='auto')
alice, bob = sf.PYU('alice'), sf.PYU('bob')

spu = sf.SPU(cluster_def=cluster_def_2)

# ------------------ distributed sys testing -----------------
# Load the data
# x1, _ = alice(breast_cancer)(party_id=1)
# x2, y = bob(breast_cancer)(party_id=2)
x1, _ = alice(breast_cancer)
x2, y = bob(breast_cancer)



# ----------------- Hyperparameter->SPU -------------------
device = spu
W = jnp.zeros((30,))
b = 0.0
W_, b_, x1_, x2_, y_ = (
    sf.to(device, W),
    sf.to(device, b),
    x1.to(device),
    x2.to(device),
    y.to(device),
)

# Train the model
losses, W_, b_ = device(
    fit,
    static_argnames=['epochs'],
    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=3,
)(W_, b_, x1_, x2_, y_, epochs=100, learning_rate=1e-2)

# TODO: how to save the output?


# --------- 测试模式的时候可以进行check ------------
# because of the sf is not instantiated in this kind of system
# print(losses)
# print(W_)
# print(b_)


# Plot the loss
# sf.reveal 将任何 DeviceObject 转换为 Python object
# has the risk to expore
# losses = sf.reveal(losses)
# plot_losses(losses)

# Validate the model
# X_test, y_test = breast_cancer(train=False)
# auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
# print(f'auc={auc}')