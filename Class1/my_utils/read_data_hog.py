import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure, color

# rgb三通道图转灰度图
def rgb2gray(img):
  gray = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140
  return gray

# 从一维rgb数组重构32*32的3通道图片
def getPhoto(pixel):
  assert len(pixel) == 3072
  r = pixel[0: 1024]; r = np.reshape(r, [32, 32, 1])
  g = pixel[1024: 2048]; g = np.reshape(g, [32, 32, 1])
  b = pixel[2048:3072]; b = np.reshape(b, [32, 32, 1])
  photo = np.concatenate([r, g, b], -1)
  photo = rgb2gray(photo)
  return photo

# 从原始数据获得hog特征
def getHog(img):
  img = getPhoto(img)
  fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
          cells_per_block=(1, 1), visualise=True)
  return fd

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as fo:
    datadict = pickle.load(fo, encoding = 'latin1')
    X = datadict['data']
    X_hog = [0] * 10000
    Y = datadict['labels']
    for i in range(len(X)):
      X_hog[i] = getHog(X[i])
    # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X_hog, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    X, Y = load_CIFAR_batch(ROOT + "data_batch_" + str(b))
    xs.append(X)
    ys.append(Y)
    print('----suc to load data_batch_' + str(b) + '----')
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(ROOT + 'test_batch')
  Xte = np.concatenate(Xte)
  return Xtr, Ytr, Xte, Yte
