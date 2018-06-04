"""
Created on 2018-06-04
# =============================================================================
# AutoEncoder training algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

image_border = 32
image_shape = (image_border, image_border)
img_rows, img_cols = image_shape

X = pd.read_csv('nepali_images_train.csv', header=None)
y = pd.read_csv('nepali_labels_train.csv', header=None)
X_test = pd.read_csv('nepali_images_test.csv', header=None)
y_test = pd.read_csv('nepali_labels_test.csv', header=None)
models = pd.read_csv('nepali_models.csv', header=None)

X = np.array(X).reshape(len(X), len(X.columns))
y = np.array(y).reshape(len(y), len(y.columns))
X_test = np.array(X_test).reshape(len(X_test), len(X_test.columns))
y_test = np.array(y_test).reshape(len(y_test), len(y_test.columns))
models = np.array(models).reshape(len(models), len(models.columns))

X = X.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
models = models.astype('float32') / 255.
X = np.reshape(X, (len(X), image_border, image_border, 1))
X_test = np.reshape(X_test, (len(X_test), image_border, image_border, 1))
models = np.reshape(models, (len(models), image_border, image_border, 1))


X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=0)

arr = np.zeros((len(X_train), image_border, image_border, 1))
for i, y in enumerate(y_train):
    arr[i, :, :, :] = models[y[0], :, :, :]
models_train = arr

arr = np.zeros((len(X_valid), image_border, image_border, 1))
for i, y in enumerate(y_valid):
    arr[i, :, :, :] = models[y[0], :, :, :]
models_valid = arr

arr = np.zeros((len(X_test), image_border, image_border, 1))
for i, y in enumerate(y_test):
    arr[i, :, :, :] = models[y[0], :, :, :]
models_test = arr

del arr


def train_model():
    input_img = Input(shape=(image_border, image_border, 1))
    # layer shape 32 x 32
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # layer shape 16 x 16
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # layer shape 8 x 8
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
    # layer shape 4 x 4
    # at this point the representation is (4, 4, 64) i.e. 572-dimensional
    x = UpSampling2D((2, 2))(encoded)
    # layer shape 8 x 8
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # layer shape 16 x 16
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # layer shape 32 x 32
    decoded = Conv2D(1, (9, 9), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train, models_train,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_valid, models_valid))

    autoencoder.save('Devanagari_autoencoder.h5')
    score = autoencoder.evaluate(X_test, models_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


train_model()