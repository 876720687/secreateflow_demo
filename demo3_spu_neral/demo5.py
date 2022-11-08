# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 19:43 
# @Author : YeMeng 
# @File : demo5.py 
# @contact: 876720687@qq.com

# 使用 JAX/FLAX 训练模型
# Load the data
from demo3_spu_neral.demo1_dataloader import *
from demo3_spu_neral.demo3_train import *
from demo3_spu_neral.demo4_validate import *

x1, _ = breast_cancer(party_id=1, train=True)
x2, y = breast_cancer(party_id=2, train=True)


# Hyperparameter
n_batch = 10
n_epochs = 10
step_size = 0.01


# Train the model
init_params = model_init(n_batch)
params = train_auto_grad(x1, x2, y, init_params, n_batch, n_epochs, step_size)

# Test the model
X_test, y_test = breast_cancer(train=False)
auc = validate_model(params, X_test, y_test)
print(f'auc={auc}')