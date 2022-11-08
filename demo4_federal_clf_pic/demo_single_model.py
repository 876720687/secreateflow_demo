# -*- coding: utf-8 -*- 
# @Time : 2022/11/3 10:35 
# @Author : YeMeng 
# @File : demo_single_model.py 
# @contact: 876720687@qq.com

from tensorflow import keras
from tensorflow.keras import layers
from demo4_federal_clf_pic.demo_federal import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    return model


alice_x = image[:10000]
alice_y = label[:10000]
alice_y = OneHotEncoder(sparse=False).fit_transform(alice_y.reshape(-1, 1))

random_seed = 1234
alice_X_train, alice_X_test, alice_y_train, alice_y_test = train_test_split(alice_x,
                                                                            alice_y,
                                                                            test_size=0.33,
                                                                            random_state=random_seed)

single_model = create_model()
single_model.fit(alice_X_train, alice_y_train, validation_data=(alice_X_test, alice_y_test), batch_size=128, epochs=10)
