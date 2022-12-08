# -*- coding: utf-8 -*- 
# @Time : 2022/12/6 14:06 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com

# TODO: make sure you have enough computation resources.
# success

import secretflow as sf
import networkx as nx
import numpy as np
import os
import pickle
import scipy
import zipfile
import tempfile
from pathlib import Path
from secretflow.utils.simulation.datasets import dataset
from secretflow.data.ndarray import load
from secretflow.utils.simulation.datasets import load_cora
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Dropout, Layer, LeakyReLU

# In case you got a running secetflow runtime already.
sf.shutdown()
sf.init(parties=['alice', 'bob'])
alice, bob = sf.PYU('alice'), sf.PYU('bob')

(edge, features, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test) = load_cora(
    [alice, bob]
)

class GraphAttention(Layer):
    def __init__(
        self,
        F_,
        attn_heads=1,
        attn_heads_reduction='average',  # {'concat', 'average'}
        dropout_rate=0.5,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        attn_kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs,
    ):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(F, self.F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel_{}'.format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.F_,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name='bias_{}'.format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name='attn_kernel_self_{}'.format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name='attn_kernel_neigh_{}'.format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(
                features, attention_kernel[0]
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(
                features, attention_kernel[1]
            )  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(
                attn_for_neighs
            )  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'attn_heads': self.attn_heads,
                'attn_heads_reduction': self.attn_heads_reduction,
                'F_': self.F_,
            }
        )
        return config


class ServerNet(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel: int,
        hidden_size: int,
        num_layer: int,
        num_class: int,
        dropout: float,
        **kwargs,
    ):
        super(ServerNet, self).__init__()
        self.num_class = num_class
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.dropout = dropout
        self.layers = []
        super(ServerNet, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers.append(
            tf.keras.layers.Dense(self.hidden_size, input_shape=(self.in_channel,))
        )
        for i in range(self.num_layer - 2):
            self.layers.append(
                tf.keras.layers.Dense(self.hidden_size, input_shape=(self.hidden_size,))
            )
        self.layers.append(
            tf.keras.layers.Dense(self.num_class, input_shape=(self.hidden_size,))
        )

        super(ServerNet, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = Dropout(self.dropout)(x)
        for i in range(self.num_layer):
            x = Dropout(self.dropout)(x)
            x = self.layers[i](x)

        return K.softmax(x)

    def compute_output_shape(self, input_shape):
        output_shape = self.hidden_size, self.output_dim
        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'in_channel': self.in_channel,
                'hidden_size': self.hidden_size,
                'num_layer': self.num_layer,
                'num_class': self.num_class,
                'dropout': self.dropout,
            }
        )
        return config
from tensorflow.keras.models import Model



def create_base_model(input_shape, n_hidden, l2_reg, num_heads, dropout_rate, learning_rate):
    def base_model():
        feature_input = tf.keras.Input(shape=(input_shape[1],))
        graph_input = tf.keras.Input(shape=(input_shape[0],))
        regular = tf.keras.regularizers.l2(l2_reg)
        outputs = GraphAttention(
            F_=n_hidden,
            attn_heads=num_heads,
            attn_heads_reduction='average',  # {'concat', 'average'}
            dropout_rate=dropout_rate,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            attn_kernel_initializer='glorot_uniform',
            kernel_regularizer=regular,
            bias_regularizer=None,
            attn_kernel_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            attn_kernel_constraint=None,
        )([feature_input, graph_input])
        # outputs = tf.keras.layers.Flatten()(outputs)
        model = Model(inputs=[feature_input, graph_input], outputs=outputs)
        model._name = "embed_model"
        # Compile model
        model.summary()
        metrics = ['acc']
        optimizer = tf.keras.optimizers.get(
            {
                'class_name': 'adam',
                'config': {'learning_rate': learning_rate},
            }
        )
        model.compile(
            loss='categorical_crossentropy',
            weighted_metrics=metrics,
            optimizer=optimizer,
        )
        return model

    return base_model



from tensorflow import keras
from tensorflow.keras import layers


def create_fuse_model(hidden_units, hidden_size, n_classes, layer_num, learning_rate):
    def fuse_model():
        inputs = [keras.Input(shape=size) for size in hidden_units]
        x = layers.concatenate(inputs)
        input_shape = x.shape[-1]
        y_pred = ServerNet(
            in_channel=input_shape,
            hidden_size=hidden_size,
            num_layer=layer_num,
            num_class=n_classes,
            dropout=0.0,
        )(x)
        # Create the model.
        model = keras.Model(inputs=inputs, outputs=y_pred, name="fuse_model")
        model.summary()
        metrics = ['acc']
        optimizer = tf.keras.optimizers.get(
            {'class_name': 'adam', 'config': {'learning_rate': learning_rate},}
        )
        model.compile(
            loss='categorical_crossentropy',
            weighted_metrics=metrics,
            optimizer=optimizer,
        )
        return model

    return fuse_model


from secretflow.ml.nn import SLModel

hidden_size = 256
n_classes = 7
attn_heads = 2
layer_num = 3
learning_rate = 1e-3
dropout_rate = 0.0
l2_reg = 0.1
num_heads = 4
epochs = 10
optimizer = 'adam'

partition_shapes = features.partition_shape()

input_shape_alice = partition_shapes[alice]
input_shape_bob = partition_shapes[bob]

sl_model = SLModel(
    base_model_dict={
        alice: create_base_model(
            input_shape_alice,
            hidden_size,
            l2_reg,
            num_heads,
            dropout_rate,
            learning_rate,
        ),
        bob: create_base_model(
            input_shape_bob,
            hidden_size,
            l2_reg,
            num_heads,
            dropout_rate,
            learning_rate,
        ),
    },
    device_y=alice,
    model_fuse=create_fuse_model(
        [hidden_size, hidden_size], hidden_size, n_classes, layer_num, learning_rate
    ),
)


sl_model.fit(
    x=[features, edge],
    y=Y_train,
    epochs=epochs,
    batch_size=input_shape_alice[0],
    sample_weight=idx_train,
    validation_data=([features, edge], Y_val, idx_val),
)


# show its performance
sl_model.evaluate(
    x=[features, edge],
    y=Y_test,
    batch_size=input_shape_alice[0],
    sample_weight=idx_test,
)