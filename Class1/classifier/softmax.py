import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        score = X[i].np.dot(W)
        score -= np.max(score) 
        correct_score = score[y[i]]
        exp_sum = np.sum(np.exp(score))
        loss += np.log(exp_sum) - correct_score
        for j in range(num_class):
            if j == y[i]:
                dW[:,j] += np.exp(score[j]) / exp_sum*X[i] - X[i]
            else:
                dW[:,j] += np.exp(score[j]) / exp_sum * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train=X.shape[0] 
    score = X.dot(W)
    score -= np.max(score, axis = 1)[:, np.newaxis]    #axis = 1每一行的最大值，score仍为500*10
    correct_score = score[range(num_train), y]    #correct_score变为500*1
    exp_score = np.exp(score)
    sum_exp_score = np.sum(exp_score, axis = 1)    #sum_exp_score为500*1
    loss = np.sum(np.log(sum_exp_score)) - np.sum(correct_score)
    exp_score /= sum_exp_score[:,np.newaxis]  #exp_score为500*10
    for i in range(num_train):
        dW += exp_score[i] * X[i][:,np.newaxis]   # X[i][:,np.newaxis]将X[i]增加一列纬度
        dW[:, y[i]] -= X[i]
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)
    dW /= num_train
    dW += reg * W
    return loss, dW