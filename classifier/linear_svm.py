import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train=X.shape[0]
    #num_classes = W.shape[1]

    scores = X.dot(W) # N * C
    margin = scores - scores[range(0, num_train), y].reshape(-1, 1) + 1 # N x C
    margin[range(num_train), y] = 0
    margin = (margin > 0) * margin
    #correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
    #margin = np.maximum(0, scores - correct_class_scores +1)
    #margins[range(num_train), list(y)] = 0

    loss += np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    # #gradient
    # margin[margin>0]=1
    # margin[margin<=0]=0

    # coeff_mat = np.zeros((num_train, num_classes))
    # coeff_mat[margins > 0] = 1
    # coeff_mat[range(num_train), list(y)] = 0
    # coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
    counts = (margin > 0).astype(int)
    counts[range(num_train), y] = - np.sum(counts, axis = 1)

    dW += (X.T).dot(counts) / num_train + reg * W
    return loss, dW