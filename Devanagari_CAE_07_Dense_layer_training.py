"""
Created on 2018-06-04
# =============================================================================
# Dense layer training algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


X = pd.read_csv('Devanagari_images_train.csv')
y = pd.read_csv('Devanagari_labels_train.csv')
X_test = pd.read_csv('Devanagari_images_test.csv')
y_test = pd.read_csv('Devanagari_labels_test.csv')

X = np.array(X).reshape(len(X), len(X.columns))
y = np.array(y).reshape(len(y), len(y.columns))
X_test = np.array(X_test).reshape(len(X_test), len(X_test.columns))
y_test = np.array(y_test).reshape(len(y_test), len(y_test.columns))

# input image dimensions
image_shape = (32, 32)
img_rows, img_cols = image_shape
nb_channels = 1

num_classes = len(np.unique(y))

model_path = 'Devanagari_dense.h5'


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
y_test_original = y_test
y_test = to_categorical(y_test)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], nb_channels, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], nb_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], nb_channels, img_rows, img_cols)
    input_shape = (64, 4, 4)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, nb_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
    input_shape = (4, 4, 64)

model = Sequential()
model.add(Dense(num_classes, activation='relu', input_shape=input_shape))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

df_encoded_train = pd.read_csv('Encoded_X_train.csv', header=None)
encoded_train = np.array(df_encoded_train)
encoded_train = encoded_train.reshape((len(df_encoded_train), ) + input_shape)
del df_encoded_train
df_encoded_valid = pd.read_csv('Encoded_X_valid.csv', header=None)
encoded_valid = np.array(df_encoded_valid)
encoded_valid = encoded_valid.reshape((len(df_encoded_valid), ) + input_shape)
del df_encoded_valid
df_encoded_test = pd.read_csv('Encoded_X_test.csv', header=None)
encoded_test = np.array(df_encoded_test)
encoded_test = encoded_test.reshape((len(df_encoded_test), ) + input_shape)
del df_encoded_test


batch_size = 32
epochs = 2
history = model.fit(encoded_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(encoded_valid, y_valid))

plot_history(history)

model.save(model_path)

predictions = model.predict(encoded_test, batch_size, verbose=1)
p('predictions', predictions)

p('y_test_original', y_test_original)
p('argmax predictions', np.argmax(predictions, axis=1))


predicted_class = np.argmax(predictions, axis=1)
p('predicted_class', predicted_class)

mask = predicted_class != np.squeeze(y_test_original)
p('mask', mask)

p('X_test', X_test)
wrong_guesses_images = X_test[mask]

wrong_guesses_target = encoded_test[mask]

wrong_guesses_predictions = predictions[mask]

wrong_guesses_class = predicted_class[mask]

good_labels = y_test_original[mask]


plot_gallery_2("Wrong guesses", wrong_guesses_images, image_shape,
               wrong_guesses_class, wrong_guesses_predictions, good_labels,
               13, 13)

plot_gallery_2("Wrong guesses target", wrong_guesses_target, image_shape,
               wrong_guesses_class, wrong_guesses_predictions, good_labels,
               13, 13)

score = model.evaluate(encoded_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
