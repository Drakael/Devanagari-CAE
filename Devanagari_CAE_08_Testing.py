"""
Created on 2018-06-04
# =============================================================================
# Devanagari CAE testing
# =============================================================================
@author: %(Drakael)s
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras import backend as K
from keras.models import load_model
# from keras.datasets import mnist


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


# input image dimensions
image_shape = (32, 32)
img_rows, img_cols = image_shape
nb_channels = 1

num_classes = 10

X_test = pd.read_csv('Devanagari_images_test.csv')
y_test = pd.read_csv('Devanagari_labels_test.csv')

X_test = np.array(X_test).reshape(len(X_test), len(X_test.columns))
y_test = np.array(y_test).reshape(len(y_test), len(y_test.columns))

X_test = X_test.reshape(len(X_test), img_rows * img_cols)
X_test = X_test.astype('float32') / 255
y_test_original = y_test
y_test = to_categorical(y_test)

if K.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], nb_channels, img_rows, img_cols)
    input_shape = (nb_channels, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
    input_shape = (img_rows, img_cols, nb_channels)

batch_size = 64
# Load previously trained autoencoder
model = load_model('Devanagari_CAE.h5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

predictions = model.predict(X_test, batch_size, verbose=1)
p('predictions', predictions)

p('y_test_original', y_test_original)
p('argmax predictions', np.argmax(predictions, axis=1))


predicted_class = np.argmax(predictions, axis=1)
p('predicted_class', predicted_class)

mask = predicted_class != np.squeeze(y_test_original)
p('mask', mask)

p('X_test', X_test)
wrong_guesses_images = X_test[mask]

wrong_guesses_predictions = predictions[mask]

wrong_guesses_class = predicted_class[mask]

good_labels = y_test_original[mask]

n_row, n_col = 30, 25


def plot_gallery_2(title, images, image_shape, predicted_class=None,
                   predictions=None, targets=None, n_col=n_col, n_row=n_row):
    p('plot_gallery_2 ' + title + ' images.shape', images.shape)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col * n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        idx_sort = np.argsort(predictions[i])[::-1]
        if predicted_class is not None and predictions is not None and targets is not None:
            first_guess = idx_sort[0]
            second_guess = idx_sort[1]
            third_guess = idx_sort[2]
            fourth_guess = idx_sort[3]
            true = targets[i]
            display = str(true)
            display += '/' + str(first_guess)
            display += '-' + str(second_guess)
            display += '-' + str(third_guess)
            display += '-' + str(fourth_guess)
            plt.title(display)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)


plot_gallery_2("Wrong guesses", wrong_guesses_images, image_shape,
               wrong_guesses_class, wrong_guesses_predictions, good_labels,
               15, 15)

score = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

weights = np.array(model.get_weights())
for w in weights:
    print(w.shape)


def plot_gallery(title, images, image_shape):
    p('plot_gallery ' + title + ' images.shape', images.shape)
    n_col = int(np.ceil(np.sqrt(images.shape[0])))
    n_row = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col * n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


def plot_layer_kernel_weights(weights, id_layer, title='Kernel weights'):
    weights = weights[0]
    weight_map = np.moveaxis(weights, [0, 1, 2, 3], [2, 3, 1, 0])
    weight_map = weight_map.reshape(-1, weights.shape[0], weights.shape[1])
    shape = (weights.shape[0], weights.shape[1])
    plot_gallery("conv1 weights", weight_map, shape)


plot_layer_kernel_weights(weights, 0, "conv1 weights")
# plot_layer_kernel_weights(weights, 2, "conv2 weights")
# plot_layer_kernel_weights(weights, 4, "conv3 weights")
