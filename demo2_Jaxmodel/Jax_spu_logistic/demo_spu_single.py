# -*- coding: utf-8 -*- 
# @Time : 2022/12/5 14:43 
# @Author : YeMeng 
# @File : demo_spu_single.py 
# @contact: 876720687@qq.com

from utils.demo_processJax import *
from utils.tools import *
# ---------------- Spu ------------------
# 伪分布式
alice, bob, spu = init_test_distributed_sys_2()
# ------------------ distributed sys testing -----------------

# Load the data
x1, _ = alice(breast_cancer)(party_id=1)
x2, y = bob(breast_cancer)(party_id=2)

# Hyperparameter->SPU
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
)(W_, b_, x1_, x2_, y_, epochs=10, learning_rate=1e-2)

# print(losses)
# print(W_)
# print(b_)


# Plot the loss
# sf.reveal 将任何 DeviceObject 转换为 Python object
# has the risk to expore
print("before {}".format(losses))
losses = sf.reveal(losses)
print("after {}".format(losses))
plot_losses(losses)


# Validate the model
X_test, y_test = breast_cancer(train=False)
auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
print(f'auc={auc}')