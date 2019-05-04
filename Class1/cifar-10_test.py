import pickle
import numpy as np
from skimage.feature import hog
from skimage import data, exposure, color
''' 数据集结构
batch_label {string}: testing batch 1 of 1
labels {list[10000]}: 每个数字取值范围0~9, 代表当前图片所属类别
data {array[10000][3072]}: 每一行代表一张图片的像素值(32*32*3=3072)
filenames {list[10000]}: 每一项是一个string代表文件名
'''

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

def train():
    for i in range(2):
        # 十种label，hog数组长度为32
        model = np.zeros((10, 32))
        label = test_lists[b'labels'][i]
        img = test_lists[b'data'][i]
        hog = np.array(getHog(img))
        # 根据label作为model的index，进行数组叠加，最后对每个数组/10000
        model[label] += hog
        print("---label: ", label)
        print("---hog: ", hog)
        print("---model: ", model)
        
test_lists = unpickle('./cifar-10-batches-py/test_batch')
# print(getHog(test_lists[b'data'][0]))
train()
