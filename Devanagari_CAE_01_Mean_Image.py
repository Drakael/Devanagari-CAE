"""
Created on 2018-06-04
# =============================================================================
# Mean image generation algorythm in python
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


src_dir = "./Dataset/train"
dest_dir = './Mean Images/'


def save_mean_image(cur_dir, composed_img=None, dest=None):
    if dest is not None and composed_img is not None:
        if not os.path.exists(dest):
            os.makedirs(dest)
        dir_ = dest + cur_dir
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        dump = dir_ + '/mean.png'
        print('dump = ', dump)
        data = np.zeros((len(composed_img),) + composed_img[0].shape)
        for i, img in enumerate(composed_img):
            data[i, :, :] = img
        out_img = np.mean(data, axis=0)
        cv2.imwrite(dump, out_img)


def mean_image(src, dest=None):
    last_cat = None
    composed_img = None
    for subdir, dirs, files in os.walk(src):
        split = subdir.split('\\')
        cur_up_dir = None
        if len(split) > 2:
            cur_up_dir = split[-2]
        cur_dir = split[-1]
        print('cur_up_dir', cur_up_dir, 'cur_dir', cur_dir)
        if last_cat is None:
            last_cat = cur_dir
            composed_img = None
        elif last_cat != cur_dir:
            last_cat = cur_dir
            composed_img = None
        cnt = 0
        for image in files:
            if image[-4:] == '.png':
                arr = cv2.imread(subdir + '/' + image, -1)
                if composed_img is None:
                    arr = 255 - arr
                    composed_img = [arr, ]
                    cnt += 1
                else:
                    arr = 255 - arr
                    composed_img.append(arr)
                    cnt += 1
        if composed_img is not None:
            save_mean_image(cur_dir, composed_img, dest)


mean_image(src_dir, dest_dir)
