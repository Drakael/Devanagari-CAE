"""
Created on 2018-06-04
# =============================================================================
# Best Mean Image selection algorythm in python
# =============================================================================
@author: %(Drakael)s
"""
import os
import cv2
import numpy as np


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


src_dir = "./Dataset/train"
mean_dir = './Mean Images/'
dest_dir = './Best Mean/'


def best_mean_image(src, dest=None):
    last_cat = None
    for subdir, dirs, files in os.walk(src):
        split = subdir.split('\\')
        cur_up_dir = None
        if len(split) > 2:
            cur_up_dir = split[-2]
        cur_dir = split[-1]
        print('cur_up_dir', cur_up_dir, 'cur_dir', cur_dir)
        mean_img = cv2.imread(mean_dir + cur_dir + '/mean.png', -1)
        if mean_img is not None:
            mean_img = 255 - mean_img
            if last_cat is None:
                last_cat = cur_dir
            elif last_cat != cur_dir:
                last_cat = cur_dir
            cnt = 0
            min_ = 100000000000
            best_img = None
            for image in files:
                if image[-4:] == '.png':
                    arr = cv2.imread(subdir + '/' + image, -1)
                    sum_ = mse(arr, mean_img)
                    if sum_ < min_:
                        min_ = sum_
                        best_img = arr
                    cnt += 1
            if dest is not None:
                dir_ = dest + cur_dir
                if not os.path.exists(dest):
                    os.makedirs(dest)
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                dump = dir_ + '/best.png'
                print('dump = ', dump)
                best_img = 255 - best_img
                cv2.imwrite(dump, best_img)


best_mean_image(src_dir, dest_dir)
