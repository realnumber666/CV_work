import numpy as np
import pickle
import matplotlib.pyplot as plt
from classifier.knn import KNearestNeighbor

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as fo:
    datadict = pickle.load(fo, encoding = 'latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    X, Y = load_CIFAR_batch(cifar10_dir + "data_batch_" + str(b))
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(cifar10_dir + 'test_batch')
  return Xtr, Ytr, Xte, Yte 


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
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

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