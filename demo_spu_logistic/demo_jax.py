# -*- coding: utf-8 -*- 
# @Time : 2022/11/1 13:28 
# @Author : YeMeng 
# @File : demo2_spu.py 
# @contact: 876720687@qq.com

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import jax.numpy as jnp
from jax import value_and_grad


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


# ----------------- define the model ----------------

# define the loss function
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.mean(jnp.log(label_probs))


def train_step(W, b, x1, x2, y, learning_rate):
    x = jnp.concatenate([x1, x2], axis=1)
    loss_value, Wb_grad = value_and_grad(loss, (0, 1))(W, b, x, y)
    W -= learning_rate * Wb_grad[0]
    b -= learning_rate * Wb_grad[1]
    return loss_value, W, b

def fit(W, b, x1, x2, y, epochs=1, learning_rate=1e-2):
    losses = jnp.array([])
    for _ in range(epochs):
        l, W, b = train_step(W, b, x1, x2, y, learning_rate=learning_rate)
        losses = jnp.append(losses, l)
    return losses, W, b


# ------------------ model validation ---------------
def validate_model(W, b, X_test, y_test):
    y_pred = predict(W, b, X_test)
    return roc_auc_score(y_test, y_pred)


def plot_losses(losses):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()