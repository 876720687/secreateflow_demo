# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 15:51 
# @Author : YeMeng 
# @File : main_process.py 
# @contact: 876720687@qq.com
# Load the data
from demo_spu_logistic.demo_jax import *

# Load the data
x1, _ = breast_cancer(party_id=1,train=True)
x2, y = breast_cancer(party_id=2,train=True)

# Hyperparameter
W = jnp.zeros((30,))
b = 0.0
epochs = 10
learning_rate = 1e-2

# Train the model
losses, W, b = fit(W, b, x1, x2, y, epochs=10, learning_rate=1e-2)

# Plot the loss
plot_losses(losses)

# Validate the model
X_test, y_test = breast_cancer(train=False)
auc=validate_model(W,b, X_test, y_test)
print(f'auc={auc}')