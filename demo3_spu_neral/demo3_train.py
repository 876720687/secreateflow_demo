# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 19:39 
# @Author : YeMeng 
# @File : demo3.py 
# @contact: 876720687@qq.com

import jax.numpy as jnp
from demo3_spu_neral.demo2_model import *
import jax


def predict(params, x):
    # TODO(junfeng): investigate why need to have a duplicated definition in notebook,
    # which is not the case in a normal python program.
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

    return MLP(FEATURES).apply(params, x)


def loss_func(params, x, y):
    pred = predict(params, x)

    def mse(y, pred):
        def squared_error(y, y_pred):
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0

        return jnp.mean(squared_error(y, pred))

    return mse(y, pred)


def train_auto_grad(x1, x2, y, params, n_batch=10, n_epochs=10, step_size=0.01):
    x = jnp.concatenate((x1, x2), axis=1)
    xs = jnp.array_split(x, len(x) / n_batch, axis=0)
    ys = jnp.array_split(y, len(y) / n_batch, axis=0)

    def body_fun(_, loop_carry):
        params = loop_carry
        for (x, y) in zip(xs, ys):
            _, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, params, grads
            )
        return params

    params = jax.lax.fori_loop(0, n_epochs, body_fun, params)
    return params


def model_init(n_batch=10):
    model = MLP(FEATURES)
    return model.init(jax.random.PRNGKey(1), jnp.ones((n_batch, FEATURES[0])))