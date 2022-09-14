import sys
import numpy as np
import numpy.matlib
import statistics as stat
import pickle
import time
import matplotlib.pyplot as plt
import tqdm
import math

def softmax(X):
    np.seterr(invalid='ignore')
    result = np.exp(X) / np.sum(np.exp(X), axis=0)
    return result

def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('/Users/yuhu/Desktop/p4-deepLearning/Deeplearning/A1/cifar-10-batches-py/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        '''
		for key, val in dict_.items(): print(key)
		... 
		b'batch_label'
		b'labels'
		b'data'
		b'filenames'
            '''
    # print("test",type(dict))
    data = dict[b'data']
    labels = dict[b'labels']
    return data.T, labels


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    # X = X[1:20,1]
    # Y = Y[1:20,1]
    # W = W[:,1:20]

    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i][0] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2-c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]


def ComputeGradients(X, Y, W, b, lamda, svm=False):
    if not svm:
        #forward pass
        P = EvaluateClassifier(X, W, b)
        #backward pass
        G = P-Y
        assert (X.shape[1] == G.shape[1])
        grad_W = np.dot(G, X.T)/X.shape[1]+2*lamda*W
        grad_b = np.dot(G, np.ones((X.shape[1], 1)))/X.shape[1]
        return [grad_W, grad_b]


def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


def ComputeCost(X, Y, W, b, lamda):
    # return output: sum of the loss
    r = lamda*np.sum(W**2)
    P = EvaluateClassifier(X,W,b)
    j_sum = np.sum(-np.log(np.sum(Y * P, axis=0))) / X.shape[1]
    j_sum = j_sum+r
    return j_sum


def ComputeAccuracy(X, y, W, b):
    acc = 0
    P_all = EvaluateClassifier(X, W, b)
    for i in range(X.shape[1]):
        P = P_all[:, i]
        predictLabel = np.argmax(P)
        if predictLabel == y[i]:
            acc += 1
    acc /= X.shape[1]
    return acc


def loadData():
    # print("Start to load data")
    filename = ["data_batch_1", "data_batch_2", "data_batch_3"]
    trainX, train_y = LoadBatch(filename[0])
    val_X, val_y = LoadBatch(filename[1])
    testX, test_y = LoadBatch(filename[2])
    trainX = preprocess(trainX)
    return [trainX, train_y, val_X, val_y, testX, test_y]


def preprocess(data):
    # print("Start to preprocess data")
    mean_X = np.mean(data, axis=1)
    mean_X = mean_X.reshape((mean_X.shape[0], 1))
    std_X = np.std(data, axis=1)
    std_X = std_X.reshape((std_X.shape[0], 1))
    # Normalize the data
    data = data - np.matlib.repmat(mean_X, 1, data.shape[1])
    data = data / np.matlib.repmat(std_X, 1, data.shape[1])
    return data


def Initialwb(X, Y):
    std = np.sqrt(2/X.shape[0])
    np.random.seed(100)
    w = std*np.random.randn(Y.shape[0], X.shape[0])
    np.random.seed(100)
    b = std*np.random.randn(Y.shape[0], 1)
    return w, b


def EvaluateClassifier(X, W, b):
    # P contains the probability for each label for the image
    # X.P has size Kxn
    assert (X.shape[0] == W.shape[1])
    s = np.dot(W, X)+b
    p = softmax(s)
    return p


def smallbatches(X, y, size, shuffle=False):
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for i in range(0, X.shape[0]-size+1, size):
        if shuffle:
            choosen = indices[i:i+size]
        else:
            choosen = slice(i, i+size)
        return X[choosen], y[choosen]


def MiniBatchGD(X, Y, GDparams, W, b, lamda):
    Wstar = W - GDparams[1]*ComputeGradients(X, Y, W, b, lamda, svm=False)[0]
    bstar = b - GDparams[1]*ComputeGradients(X, Y, W, b, lamda, svm=False)[1]
    return Wstar, bstar


if __name__ == "__main__":
    lamda = 0
    eps = sys.float_info.epsilon
    #batch size
    n_batch = 100
    #eta: learning rate
    eta = 0.001
    n_epochs = 40
    GDparams = [n_batch, eta, n_epochs]
    trainX, trainy, valX, valy, testX, testY = loadData()
    trainY = np.eye(10, dtype=int)[trainy].T
    valY = np.eye(10, dtype=int)[valy].T

    W, b = Initialwb(trainX, trainY)
    # trainX_batch, trainY_batch = smallbatches(trainX.T, trainY, n_batch)
    # trainX_batch = trainX_batch.T
    # trainY_batch = np.eye(10, dtype=int)[trainY_batch].T
    # P = EvaluateClassifier(trainX_batch, W, b)
    # print(P)

    # gnw, gnb = ComputeGradsNum(
    #     trainX_batch, trainY_batch, P, W, b, lamda,h )
    # gaw, gab = ComputeGradients(
    #     trainX_batch, trainY_batch, W, b, lamda, svm=False)
    # errorW = np.abs(gaw-gnw)/(np.abs(gaw)+np.abs(gnw))
    # print(np.max(errorW))

    train_cost = []
    val_cost = []
    iterator = tqdm.tqdm(range(n_epochs))
    for epoch in iterator:
        #shuffle
        permutation = np.random.permutation(trainX.shape[1])
        for j in range(math.ceil(trainX.shape[1]/n_batch)):
            index = permutation[j*n_batch:min((j+1)*n_batch,trainX.shape[1]-1)]
            Xbatch = trainX[:,index]
            Ybatch = trainY[:,index]
            W, b = MiniBatchGD(Xbatch, Ybatch, GDparams, W, b, lamda)
        
        train_acc = ComputeAccuracy(trainX, trainy, W, b)*100
        val_acc = ComputeAccuracy(valX,valy,W,b)*100
        iterator.set_description(
                "Epoch: {} |Training acc: {:.1f}% | Validation acc: {:.1f}%".format(epoch, train_acc, val_acc))
        
        #plot 
        train_cost.append(ComputeCost(trainX, trainY, W, b,lamda))
        val_cost.append(ComputeCost(valX, valY, W, b,lamda))
    plt.plot(train_cost,color = 'r',label = "train cost")
    plt.plot(val_cost,color = 'r',label = "val cost")
    plt.savefig("train_loss.png")
    plt.savefig("val_loss.png")

    

        

