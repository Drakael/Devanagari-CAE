"""
Created on 2018-06-04
# =============================================================================
# Encoding data algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import load_model
from keras.models import Model


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


X = pd.read_csv('Devanagari_images_train.csv')
y = pd.read_csv('Devanagari_labels_train.csv')
X_test = pd.read_csv('Devanagari_images_test.csv')

X = np.array(X).reshape(len(X), len(X.columns))
y = np.array(y).reshape(len(y), len(y.columns))
X_test = np.array(X_test).reshape(len(X_test), len(X_test.columns))

# input image dimensions
image_shape = (32, 32)
img_rows, img_cols = image_shape
nb_channels = 1

num_classes = len(np.unique(y))


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,
                                                      random_state=0)

X_train = X_train.reshape(len(X_train), img_rows * img_cols)
X_train = X_train.astype('float32') / 255
X_valid = X_valid.reshape(len(X_valid), img_rows * img_cols)
X_valid = X_valid.astype('float32') / 255
X_valid_original = X_valid
y_train = to_categorical(y_train)
y_valid_original = y_valid
y_valid = to_categorical(y_valid)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], nb_channels, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], nb_channels, img_rows, img_cols)
    input_shape = (nb_channels, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, nb_channels)
    input_shape = (img_rows, img_cols, nb_channels)


batch_size = 64
# Load previously trained autoencoder
autoencoder = load_model('Devanagari_autoencoder.h5')

encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer('encoder').output)

encoder_train = encoder.predict(X_train, batch_size=batch_size, verbose=1)
encoder_valid = encoder.predict(X_valid, batch_size=batch_size, verbose=1)
encoder_test = encoder.predict(X_test, batch_size=batch_size, verbose=1)

encoder_train = pd.DataFrame(encoder_train.reshape(X_train.shape[0], -1))
encoder_train.to_csv('Encoded_X_train.csv', header=False, index=False)
del encoder_train
encoder_valid = pd.DataFrame(encoder_valid.reshape(X_valid.shape[0], -1))
encoder_valid.to_csv('Encoded_X_valid.csv', header=False, index=False)
del encoder_valid
encoder_test = pd.DataFrame(encoder_test.reshape(X_test.shape[0], -1))
encoder_test.to_csv('Encoded_X_test.csv', header=False, index=False)
del encoder_test
