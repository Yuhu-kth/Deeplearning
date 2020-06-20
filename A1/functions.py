import numpy as np
import numpy.matlib
import statistics as stat
import pickle
import time
import matplotlib.pyplot as plt

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('/Users/yuhu/Desktop/p4-deepLearning/A1/cifar-10-batches-py/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	print("test",type(dict))
	data = dict[b'data']
	labels = dict[b'labels']
	return data.T, labels

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = ComputeCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

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
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def ComputeCost(X, Y, W, b, lamda):
	return output

def Data():
	filename = ["data_batch_1","data_batch_2","data_batch_3"]
	trainX,train_Labels= LoadBatch(filename[0])
	v_X,v_Labels= LoadBatch(filename[1])
	testX,test_Labels= LoadBatch(filename[2])
	print(trainX.shape)
	preprocess(trainX)

def preprocess(data):
	mean_X = np.mean(data,axis=1)
	mean_X = mean_X.reshape((mean_X.shape[0],1))
	std_X = np.std(data,axis=1)
	std_X = std_X.reshape((std_X.shape[0],1))

	# data = data - np.tile(mean_X,(1,data.shape[0]))
	data =  data - np.matlib.repmat(mean_X,1,data.shape[1])
	start_time = time.time()
	# data =  np.divide(data,np.matlib.repmat(std_X,1,data.shape[1]))
	data =  data / np.matlib.repmat(std_X,1,data.shape[1])

	print(mean_X.shape)
	print(std_X.shape)
	# print(time.time() - start_time)#0.3113

def InitialW(k,d):
	w = np.random.normal(0,0.01,size=(k,d))

def bias(k):
	b = np.random.normal(0,0.01,size=(k,1))



if __name__ == "__main__":
	Data()







