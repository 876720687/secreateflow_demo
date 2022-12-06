# -*- coding: utf-8 -*- 
# @Time : 2022/11/3 10:18 
# @Author : YeMeng 
# @File : demo.py 
# @contact: 876720687@qq.com
import numpy as np
import secretflow as sf
from matplotlib import pyplot as plt
from secretflow.ml.nn import FLModel
from secretflow.security import SecureAggregator, DeviceAggregator
from secretflow.utils.simulation.datasets import load_mnist, dataset

# In case you have a running secretflow runtime already.
# sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], num_cpus=8, log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))


(x_train, y_train), (x_test, y_test) = load_mnist(parts=[alice, bob], normalized_x=True, categorical_y=True)

mnist = np.load(dataset('mnist'), allow_pickle=True)
image = mnist['x_train']
label = mnist['y_train']


def create_conv_model(input_shape, num_classes, name='model'):
  def create_model():
    from tensorflow import keras
    from tensorflow.keras import layers
    # Create model
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

  return create_model

# ----------------- model training -----------------
num_classes = 10
input_shape = (28, 28, 1)
model = create_conv_model(input_shape, num_classes)
device_list = [alice, bob]

secure_aggregator = SecureAggregator(charlie, [alice, bob])
spu_aggregator = DeviceAggregator(spu)
fed_model = FLModel(server=charlie,
                    device_list=device_list,
                    model=model,
                    aggregator=secure_aggregator,
                    strategy="fed_avg_w",
                    backend="tensorflow")

history = fed_model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        sampler_method="batch",
                        batch_size=128,
                        aggregate_freq=1)

# Draw accuracy values for training & validation
plt.plot(history.global_history['accuracy'])
plt.plot(history.global_history['val_accuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# Draw loss for training & validation
plt.plot(history.global_history['loss'])
plt.plot(history.global_history['val_loss'])
plt.title('FLModel loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

global_metric = fed_model.evaluate(x_test, y_test, batch_size=128)
print(global_metric)