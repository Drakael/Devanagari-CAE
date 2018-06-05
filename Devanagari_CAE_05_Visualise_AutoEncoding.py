import numpy as np
import cv2
from keras.models import load_model
import time
import pandas as pd

print('Loading Devanagari dataset')
t0 = time.time()

X_test = pd.read_csv('Devanagari_images_test.csv', header=None)
t1 = time.time()
print('Devanagari dataset loaded in: ', t1 - t0)

X_test = np.array(X_test).reshape(len(X_test), len(X_test.columns))
np.random.shuffle(X_test)

image_border_size = 32
image_border_size_ten = image_border_size * 10
image_shape = (image_border_size, image_border_size)
size_ten = (image_border_size_ten, image_border_size_ten)
img_rows, img_cols = image_shape


X_test = X_test.astype('float32') / 255.
X_test = np.reshape(X_test,
                    (len(X_test), image_border_size, image_border_size, 1))

print('Loading model :')
t0 = time.time()
# Load previously trained autoencoder
autoencoder = load_model('Devanagari_autoencoder.h5')
t1 = time.time()
print('Model loaded in: ', t1 - t0)


def plot_encoded_images(model, data, encoded_images, idx, shape):
    resized_test_img = cv2.resize(data[idx], shape)
    cv2.imshow('input', resized_test_img)
    cv2.waitKey(0)
    resized_output = cv2.resize(encoded_images[idx], shape)
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)

samples = X_test[0:20]
encoded_images = autoencoder.predict(samples, verbose=1)
for i in range(20):
    plot_encoded_images(autoencoder, samples, encoded_images, i, size_ten)
