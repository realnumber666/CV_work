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
    X, Y = load_CIFAR_batch(cifar10_dir + "data_batch_" + str(b))
    xs.append(X)
    ys.append(Y)
    print('----suc to load data_batch_' + str(b) + '----')
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(cifar10_dir + 'test_batch')
  Xte = np.concatenate(Xte)
  return Xtr, Ytr, Xte, Yte

class KNearestNeighbor(object):
    """a KNN classifier with L2 distance"""
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. This is just memorizing all the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train,) containing the training labels, 
          where y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k = 1, num_loops = 0):
        """
        Test the classifier. 
        Inputs:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determine whether use for-loop to calculate L2 distance
          between the train points and test points
        Return:
        - pred_y: Predict output y
        """
        # calculate the L2 distance between test X and train X
        if num_loops == 0:
            # no for-loop, vectorized
            dists = self.cal_dists_no_loop(X)
        elif num_loops == 1:
            # one for-loop, half-vectorized
            dists = self.cal_dists_one_loop(X)
        elif num_loops == 2:
            # two for-loop, no vectorized
            dists = self.cal_dists_two_loop(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        # predict the labels
        num_test = X.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dists_k_min = np.argsort(dists[i])[:k]    # the closest k distance loc 
            close_y = self.y_train[dists_k_min]    # the closest k distance ,all labels
            y_pred[i] = np.argmax(np.bincount(close_y))    # [0,3,1,3,3,1] -> 3　as y_pred[i]

        return y_pred

    def cal_dists_no_loop(self, X):
        """
        Calculate the distance with no for-loop
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
        d1 = np.multiply(np.dot(X, self.X_train.T), -2)    # shape (num_test, num_train)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
        dists = np.sqrt(d1 + d2 + d3)

        return dists

    def cal_dists_one_loop(self, X):
        """
        Calculate the distance with one for-loop
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))

        return dists

    def cal_dists_two_loop(self, X):
        """
        Calculate the distance with two for-loop
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

        return dists 


cifar10_dir = "cifar-10-batches-py/"

# Load the raw CIFAR-10 data.
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir) 


# train numbers
num_train = 5000
mask = range(num_train)
X_train = X_train[mask]
y_train = y_train[mask]

# test numbers
num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# change 4D to 2D, like (5000, 32, 32, 3) -> (5000, 3072)
# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))

KNN = KNearestNeighbor()
KNN.train(X_train, y_train)

num_folds = 5    # split the training dataset to 5 parts
k_classes = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]    # all k, determine the best k

# Split up the training data into folds
X_train_folds = []
y_train_folds = []
X_train_folds = np.split(X_train, num_folds)
y_train_folds = np.split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k
k_accuracy = {}

for k in k_classes:
    accuracies = []
    #knn = KNearestNeighbor()
    for i in range(num_folds):
        Xtr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        Xcv = X_train_folds[i]
        ycv = y_train_folds[i]
        KNN.train(Xtr, ytr)
        ycv_pred = KNN.predict(Xcv, k=k, num_loops=0)
        accuracy = np.mean(ycv_pred == ycv)
        accuracies.append(accuracy)
    k_accuracy[k] = accuracies

# Print the accuracy
for k in k_classes:
    for i in range(num_folds):
        print('k = %d, fold = %d, accuracy: %f' % (k, i+1, k_accuracy[k][i]))

#Plot the cross validation
for k in k_classes:
    plt.scatter([k] * num_folds, k_accuracy[k])

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = [np.mean(k_accuracy[k]) for k in k_accuracy]
accuracies_std = [np.std(k_accuracy[k]) for k in k_accuracy]
plt.errorbar(k_classes, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Choose the best k
best_k = k_classes[np.argmax(accuracies_mean)]
# Use the best k, and test it on the test data
KNN = KNearestNeighbor()
KNN.train(X_train, y_train)
y_pred = KNN.predict(X_test, k=best_k, num_loops=0)
num_correct = np.sum(y_pred == y_test)
accuracy = np.mean(y_pred == y_test)
print('Correct %d/%d: The accuracy is %f' % (num_correct, X_test.shape[0], accuracy))