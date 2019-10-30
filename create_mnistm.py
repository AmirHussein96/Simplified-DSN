import tarfile
import os
import hickle as hkl
import numpy as np
import pandas as pd
import skimage
import skimage.io
import skimage.transform
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

BST_PATH = 'BSR_bsds500.tgz'

rand = np.random.RandomState(42)

f = tarfile.open(BST_PATH)
train_files = []
for name in f.getnames():
    if name.startswith('BSR/BSDS500/data/images/train/'):
        train_files.append(name)

print
'Loading BSR training images'
background_data = []
for name in train_files:
    try:
        fp = f.extractfile(name)
        bg_img = skimage.io.imread(fp)
        background_data.append(bg_img)
    except:
        continue


def compose_image(digit, background):
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x + dw, y:y + dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        if i % 1000 == 0:
            print
            i
        bg_img = rand.choice(background_data)
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_

def savefile(history,path):
#    if not os.path.exists(path):
#        os.makedirs(path)
    hkl.dump(history,path)
path='data\\'    
print
'Building train set...'
train = create_mnistm(mnist.train.images)
print
'Building test set...'
test = create_mnistm(mnist.test.images)
print
'Building validation set...'
valid = create_mnistm(mnist.validation.images)

mnist_data={'train':train,'test':test,'valid':valid}
# Save dataset as pickle
path = os.path.join(str(path), 'mnistm_data.hkl')
savefile( mnist_data,path )

