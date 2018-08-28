#97% Accuracy on 2 epoch

from keras.datasets import mnist
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import normalize
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

model_Name = "Mnist_Conv_batchNormalise"
tensorboard_name = 'conv(32-64)_batchNormalise/'

#Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalise the data
x_train = normalize(x_train, axis=-1)
x_test = normalize(x_test, axis=-1)

#Reshape Data
#2-for conv
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28,28, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28, 28, 1), data_format="channels_last"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Tensorboard for visualise
tensorboard = TensorBoard(log_dir='Mnist_log/' + tensorboard_name,
                          histogram_freq=30)

#Feed the data
model.fit(x_train, y_train, epochs=2, batch_size=128,
          validation_data=(x_test, y_test), callbacks=[tensorboard])

#Save Model
model.save(model_Name + '.model')

#Delete existing model
del model

#load saved model
save_model = keras.models.load_model(model_Name + '.model')


