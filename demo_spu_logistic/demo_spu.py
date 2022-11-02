# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 18:41 
# @Author : YeMeng 
# @File : demo_spu.py 
# @contact: 876720687@qq.com
import secretflow as sf
from demo_spu_logistic.demo_jax import *

# sf.shutdown()

sf.init(['alice', 'bob'], num_cpus=8, log_to_driver=True)

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

x1, _ = alice(breast_cancer)(party_id=1)
x2, y = bob(breast_cancer)(party_id=2)

# 将超参数和所有数据传递给 SPU 设备
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

# ---------------- model train ----------------
# 运算出来的模型和结果也是保密状态
losses, W_, b_ = device(
    fit,
    static_argnames=['epochs'],
    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=3,
)(W_, b_, x1_, x2_, y_, epochs=10, learning_rate=1e-2)

print(losses)
print(W_)
print(b_)

# ---------------- model train ----------------
# sf.reveal 将任何 DeviceObject 转换为 Python object
losses = sf.reveal(losses)
# plot_losses(losses)
X_test, y_test = breast_cancer(train=False)
auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
print(f'auc={auc}')
