"""
Created on 2018-06-04
# =============================================================================
# CSV processing algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

image_border = 32
square_size = image_border**2


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


def feed_from_dir(src_dir, reverse=False):
    output_x = []
    output_y = []
    last_char = None
    idx_char = 0
    for subdir, dirs, files in os.walk(src_dir):
        split = subdir.split('\\')
        cur_up_dir = None
        if len(split) > 2:
            cur_up_dir = split[-2]
        cur_dir = split[-1]
        print('cur_up_dir', cur_up_dir, 'cur_dir', cur_dir)
        for image in files:
            if image[-4:] == '.png':
                if last_char is None:
                    last_char = cur_dir
                elif last_char != cur_dir:
                    idx_char += 1
                    last_char = cur_dir
                    print('idx_char', idx_char)
                image_path = subdir + '/' + image
                # print('image_path', image_path)
                image_array = imageio.imread(image_path)
                if reverse is True:
                    image_array = 255 - image_array
                row_x = image_array.reshape(1, square_size).tolist()
                row_y = idx_char
                output_x.append(row_x)
                output_y.append(row_y)
    return output_x, output_y


traindir = './Dataset/train/'
testdir = './Dataset/test/'
modeldir = './Best Mean/'

output_images_train, output_labels_train = feed_from_dir(traindir, True)
output_images_test, output_labels_test = feed_from_dir(testdir, True)
output_models, output_models_indices = feed_from_dir(modeldir)

np_array = np.array(output_images_train)
np_array = np_array.reshape(len(output_images_train), 32 * 32)
df = pd.DataFrame(np_array)
df.to_csv('Devanagari_images_train.csv', header=False, index=False)

np_array = np.array(output_models)
np_array = np_array.reshape(len(output_models), square_size)
df = pd.DataFrame(np_array)
df.to_csv('Devanagari_models.csv', header=False, index=False)


def plot_gallery_2(title, images, image_shape, n_col=10, n_row=10):
    p('plot_gallery_2 ' + title + ' images.shape', images.shape)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col * n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.title(i)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)


plot_gallery_2("Models", np_array, (image_border, image_border), 8, 8)

np_array = np.array(output_labels_train)
np_array = np_array.reshape(len(output_labels_train), 1)
df = pd.DataFrame(np_array)
df.to_csv('Devanagari_labels_train.csv', header=False, index=False)

np_array = np.array(output_images_test)
np_array = np_array.reshape(len(output_images_test), square_size)
df = pd.DataFrame(np_array)
df.to_csv('Devanagari_images_test.csv', header=False, index=False)

np_array = np.array(output_labels_test)
np_array = np_array.reshape(len(output_labels_test), 1)
df = pd.DataFrame(np_array)
df.to_csv('Devanagari_labels_test.csv', header=False, index=False)
