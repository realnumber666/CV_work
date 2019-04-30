
#%%
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)


#%%
num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# 提取tfidf特征 TODO
vectorizer = TfidfVectorizer(max_features=2000)

# 对训练和测试数据一起提取特征
X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )

# 分离出训练数据和测试数据
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target


#%%
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(len(X_train.toarray()))


#%%
def calculate_loss(model, X, y):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    #正向传播，计算预测值
    z1 = X.dot(W1) + b1
    # a1 = np.tanh(z1) # 纯tanh
    # a1 = np.maximum(0, z1) # 纯ReLU
    a1 = np.maximum(0, z1) # 混合
    z2 = a1.dot(W2) + b2
    # a2 = np.tanh(z2)
    # a2 = np.maximum(0, z2)
    a2 = np.maximum(0, z2)
    z3 = a2.dot(W3) + b3
    # a3 = np.tanh(z3)
    # a3 = np.maximum(0, z3)
    a3 = np.maximum(0, z3)
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #在损失上加上正则项（可选）
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)) + np.sum(np.square(W4)))
    return 1./num_examples * data_loss


#%%
# Helper function to predict an output (0 or 1)
# sigmoid : # a3 = 1./(1 + np.exp(-z3))
def predict(model, X):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    # 正向传播
    z1 = X.dot(W1) + b1
    # a1 = np.tanh(z1) # 纯tanh
    # a1 = np.maximum(0, z1) # 纯ReLU
    a1 = np.maximum(0, z1) # 混合
    z2 = a1.dot(W2) + b2
    # a2 = np.tanh(z2)
    # a2 = np.maximum(0, z2)
    a2 = np.maximum(0, z2)
    z3 = a2.dot(W3) + b3
    # a3 = np.tanh(z3)
    # a3 = np.maximum(0, z3)
    a3 = np.maximum(0, z3)
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


#%%
# 这个函数为神经网络学习参数并且返回模型
# - nn_hdim: 隐藏层的节点数
# - num_passes: 通过训练集进行梯度下降的次数
# - print_loss: 如果是True, 那么每1000次迭代就打印一次损失值
def build_model(X, y, nn_hdim, epsilon, reg_lambda, num_passes=20000,  print_loss=False):
    
    # 用随机数初始化参数
    np.random.seed(0)
    W1 = np.random.randn(input_dim, nn_hdim[0]) / np.sqrt(input_dim)
    b1 = np.zeros((1, nn_hdim[0]))
    W2 = np.random.randn(nn_hdim[0],nn_hdim[1]) / np.sqrt(nn_hdim[0])
    b2 = np.zeros((1, nn_hdim[1]))
    W3 = np.random.randn(nn_hdim[1],nn_hdim[2]) / np.sqrt(nn_hdim[1])
    b3 = np.zeros((1, nn_hdim[2]))
    W4 = np.random.randn(nn_hdim[2], np.shape(categories)[0]) / np.sqrt(nn_hdim[2])
    b4 = np.zeros((1, np.shape(categories)[0]))

    model = {}
    # 梯度下降
    for i in range(0, num_passes):
        # - learning rate 选择策略 -
        epsilon_change = epsilon # fixed
        # epsilon_change = epsilon * np.power(0.995,i) # exp
        # epsilon_change = epsilon * np.power(0.98,np.floor(i/100)) # step
        # epsilon_change = epsilon * np.power((1+0.001*i), -1) # inv

        # 正向传播，计算判断出的结果
        # # - 纯tanh -
        # z1 = X.dot(W1) + b1
        # d1 = (np.random.rand(z1.shape[0],z1.shape[1])) < keep_rate
        # a1 = (np.tanh(z1) * d1)/keep_rate
        # z2 = a1.dot(W2) + b2
        # d2 = (np.random.rand(z2.shape[0],z2.shape[1])) < keep_rate
        # a2 = (np.tanh(z2) * d2)/keep_rate
        # z3 = a2.dot(W3) + b3
        # d3 = (np.random.rand(z3.shape[0],z3.shape[1])) < keep_rate
        # a3 = (np.tanh(z3) * d3)/keep_rate
        # z4 = a3.dot(W4) + b4

        # # - 纯ReLU -
        # z1 = X.dot(W1) + b1
        # d1 = (np.random.rand(z1.shape[0],z1.shape[1])) < keep_rate
        # a1 = (np.maximum(0, z1) * d1)/keep_rate
        # z2 = a1.dot(W2) + b2
        # d2 = (np.random.rand(z2.shape[0],z2.shape[1])) < keep_rate
        # a2 = (np.maximum(0, z2) * d2)/keep_rate
        # z3 = a2.dot(W3) + b3
        # d3 = (np.random.rand(z3.shape[0],z3.shape[1])) < keep_rate
        # a3 = (np.maximum(0, z3) * d3)/keep_rate
        # z4 = a3.dot(W4) + b4

        # - 混合 -
        z1 = X.dot(W1) + b1
        d1 = (np.random.rand(z1.shape[0],z1.shape[1])) < keep_rate
        a1 = (np.maximum(0, z1) * d1)/keep_rate #relu
        z2 = a1.dot(W2) + b2
        d2 = (np.random.rand(z2.shape[0],z2.shape[1])) < keep_rate
        a2 = (np.tanh(z2) * d2)/keep_rate #tanh
        z3 = a2.dot(W3) + b3
        d3 = (np.random.rand(z3.shape[0],z3.shape[1])) < keep_rate
        a3 = ((1./(1 + np.exp(-z3))) * d3)/keep_rate #sigmoid
        z4 = a3.dot(W4) + b4        


        exp_scores = np.exp(z4)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # 反向传播，对参数进行优化
        # TODO num_examples ?= nn_input_dim

        # - 混合激活函数 -
        delta4 = probs
        delta4[range(num_examples), y] -= 1
        dW4 = (a3.T).dot(delta4)
        db4 = np.sum(delta4, axis=0, keepdims=True)

        delta3 = delta4.dot(W4.T) * a3 * (1 - a3) * d3/keep_rate 
        dW3 = (a2.T).dot(delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W3.T) * (1 - np.power(a2, 2)) * d2/keep_rate 
        dW2 = (a1.T).dot(delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = delta2.dot(W2.T) * d1/keep_rate
        delta1[z1 <= 0] = 0 
        dW1 = (X.T).dot(delta1)
        db1 = np.sum(delta1, axis=0)

        # # - 纯tanh -
        # delta4 = probs
        # delta4[range(num_examples), y] -= 1
        # dW4 = (a3.T).dot(delta4)
        # db4 = np.sum(delta4, axis=0, keepdims=True)

        # delta3 = delta4.dot(W4.T) * (1 - np.power(a3, 2)) * d3/keep_rate 
        # dW3 = (a2.T).dot(delta3)
        # db3 = np.sum(delta3, axis=0, keepdims=True)
        # delta2 = delta3.dot(W3.T) * (1 - np.power(a2, 2)) * d2/keep_rate 
        # dW2 = (a1.T).dot(delta2)
        # db2 = np.sum(delta2, axis=0)
        # delta1 = delta2.dot(W2.T) * (1 - np.power(a1, 2)) * d1/keep_rate
        # dW1 = (X.T).dot(delta1)
        # db1 = np.sum(delta1, axis=0)

        # # - 纯ReLU -
        # delta4 = probs
        # delta4[range(num_examples), y] -= 1
        # dW4 = (a3.T).dot(delta4)
        # db4 = np.sum(delta4, axis=0, keepdims=True)

        # delta3 = delta4.dot(W4.T) * d3/keep_rate 
        # delta3[z3 <= 0] = 0 
        # dW3 = (a2.T).dot(delta3)
        # db3 = np.sum(delta3, axis=0, keepdims=True)
        # delta2 = delta3.dot(W3.T) * d2/keep_rate 
        # delta2[z2 <= 0] = 0 
        # dW2 = (a1.T).dot(delta2)
        # db2 = np.sum(delta2, axis=0)
        # delta1 = delta2.dot(W2.T) * d1/keep_rate
        # delta1[z1 <= 0] = 0 
        # dW1 = (X.T).dot(delta1)
        # db1 = np.sum(delta1, axis=0)
        
        # 添加正则项
        dW4 += reg_lambda * W4
        dW3 += reg_lambda * W3
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        
        # 梯度下降更新参数
        W1 += -epsilon_change * dW1
        b1 += -epsilon_change * db1
        W2 += -epsilon_change * dW2
        b2 += -epsilon_change * db2
        W3 += -epsilon_change * dW3
        b3 += -epsilon_change * db3
        W4 += -epsilon_change * dW4
        b4 += -epsilon_change * db4
        
        # 为模型分配新参数
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4 }
        
        # 选择性打印loss
        if print_loss and i % 100 == 0:
            print("Loss after iteration", i, calculate_loss(model, X, y))
    return model


#%%
num_examples, input_dim = X_train.shape # 样本数量，输入层维度（target种类数）
epsilon = 0.0001 # 梯度下降学习率
reg_lambda = 0.001 # 正则化强度
epochs = 10000 # 梯度下降的次数:1个epoch表示过了1遍训练集中的所有样本
nn_hdims = [16, 16, 8]
keep_rate = 0.8 # dropout保留程度

#%%
model = build_model(X_train, Y_train, nn_hdims, epsilon, reg_lambda, epochs, print_loss=True)


#%%
n_correct = 0
n_test = X_test.shape[0]
for n in range(n_test):
    x = X_test[n,:]
    yp = predict(model, x)
    if yp == Y_test[n]:
        n_correct += 1.0

print('Accuracy %f = %d / %d'%(n_correct/n_test, int(n_correct), n_test) )


#%%



