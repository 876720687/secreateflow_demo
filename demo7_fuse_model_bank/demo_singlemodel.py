# -*- coding: utf-8 -*- 
# @Time : 2022/12/6 10:53 
# @Author : YeMeng 
# @File : demo_singlemodel.py 
# @contact: 876720687@qq.com
from secretflow.utils.simulation.datasets import dataset
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_model():

    model = keras.Sequential(
        [
            keras.Input(shape=4),
            layers.Dense(100,activation ="relu" ),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy",tf.keras.metrics.AUC()])
    return model

single_model = create_model()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(dataset('bank_marketing'), sep=';')
alice_data = df[["age", "job", "marital", "education", "y"]]
bob_data = df[["default", "balance", "housing", "loan", "contact",
             "day","month","duration","campaign","pdays","previous","poutcome"]]

encoder = LabelEncoder()
alice_data.loc[:, 'job'] = encoder.fit_transform(alice_data['job'])
alice_data.loc[:, 'marital'] = encoder.fit_transform(alice_data['marital'])
alice_data.loc[:, 'education'] = encoder.fit_transform(alice_data['education'])
alice_data.loc[:, 'y'] =  encoder.fit_transform(alice_data['y'])

y = alice_data['y']
alice_data = alice_data.drop(columns=['y'],inplace=False)
scaler = MinMaxScaler()
alice_data = scaler.fit_transform(alice_data)
random_state = 1234
train_data,test_data = train_test_split(alice_data,train_size=0.8,random_state=random_state)
train_label,test_label = train_test_split(y,train_size=0.8,random_state=random_state)
single_model.fit(train_data,train_label,validation_data=(test_data,test_label),batch_size=128,epochs=10,shuffle=False)
