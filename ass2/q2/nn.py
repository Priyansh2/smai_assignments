import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def relu(data_array):
    return np.maximum(data_array, 0)

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def regularization_L2_softmax_loss(reg_lambda, weight1, weight2):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

file = list(open("data3.csv","r"))

x1,y1 =[],[]
for i in range(len(file)):
	if file[i].find('?') != -1:
		continue

	x = file[i].strip().split(',')[:-1]
	y = file[i].strip().split(',')[-1]
	if y <= ' 4' and y>=' 1':
		x1.append(x)
		y1.append(y)
# print len(x1[0])
X,Y =[],[]
for i in range(len(x1)):
	X.append(map(int,x1[i]))
	# del X[i][0]
X = np.array(X)
print len(X[0])
print y1[0]
for i in range(len(y1)):
	y1[i] = y1[i][1:]
	Y.append(map(int,y1[i]))
# print Y
for i in range(len(Y)):
	if Y[i]==[1]: Y[i]=0
	if Y[i]==[2]: Y[i]=1
	if Y[i]==[3]: Y[i]=2
	if Y[i]==[4]: Y[i]=3
Y = np.array(Y)

Y1 = np.zeros((X.shape[0], 4))
print Y1.shape
for i in range(X.shape[0]):
    Y1[i, Y[i]] = 1

def nerualNet(X,Y1,epochno,mod):
# temp = Y1
# print Y1.shape,temp.T.shape
# definning parameters for NN
	s = X.shape[0] #  samples
	n = X.shape[1] #  features per sample
	d = 30 # nodes in the hidden layer
	c = 4 # three classes to predict 
	# W1 = [np.zeros(d) for i in range(n)]
	# W1 = np.array(W1)
	# W2 = [np.zeros(c) for i in range(d)]
	# W2 = np.array(W2)
	W1 = np.random.normal(0, 1, [n, d]) 
	W2 = np.random.normal(0, 1, [d, c]) 

	b1 = np.zeros((1, d))
	b2 = np.zeros((1, c))

	alpha = 10e-3
	for epoch in range(epochno):
	    # forward pass
	    A = sigmoid(X.dot(W1) + b1) 
	    B = softmax(A.dot(W2) + b2)
	    
	    loss = cross_entropy_softmax_loss_array(B, Y1)
	    loss += regularization_L2_softmax_loss(0.001, W1, W2) 

	    # backward pass'
	    delta2 = (B - Y1)/(B.shape[0])
	    delta1 = ((delta2).dot(W2.T))* A * (1-A)
	    # delta1[A<=0] = 0
	    
	    gradient_W2 = np.dot(A.T,delta2) + 0.001*W2
	    gradient_W1 = np.dot(X.T,delta1) + 0.001*W1

	    W2 -= alpha * gradient_W2
	    b2 -= alpha * (delta2).sum(axis=0)

	    W1 -= alpha * gradient_W1
	    b1 -= alpha * (delta1).sum(axis=0)
	    #accuracy
	    ans=0
	    for i in range(X.shape[0]):
	    	temp = map(int,np.unravel_index(B[i].argmax(),B[i].shape))
	    	temp1= map(int,np.unravel_index(Y1[i].argmax(),Y1[i].shape))
	    	# print temp1 , temp
	    	if temp!=temp1:
	    		ans+=1
	    # print ans
	    fans = 100 - ans/float(X.shape[0])*100
	    if epoch%mod==0:
	    	print epoch,"===>",fans,"===>",loss
	return W1,W2,b1,b2    	

W1,W2,b1,b2 = nerualNet(X,Y1,8001,1000)
# print B
# plt.plot(costs)
# plt.show()
print X.shape
print W1.shape
# print A.shape
print W2.shape
# print B.shape

file1 = list(open("data2.csv","r"))
x11,y11 =[],[]
for i in range(len(file1)):
	if file1[i].find('?') != -1:
		continue

	x = file1[i].strip().split(',')[:-1]
	y = file1[i].strip().split(',')[-1]
	if y <= ' 4' and y >=' 1':
		x11.append(x)
		y11.append(y)
# print len(x1[0])
X1,Y11 =[],[]
print y11
for i in range(len(x11)):
	X1.append(map(int,x11[i]))
	# del X[i][0]
X1 = np.array(X1)
for i in range(len(y11)):
	y11[i] = y11[i][1:]
	Y11.append(map(int,y11[i]))
# print Y
for i in range(len(Y11)):
	if Y11[i]==[1]: Y11[i]=0
	if Y11[i]==[2]: Y11[i]=1
	if Y11[i]==[3]: Y11[i]=2
	if Y11[i]==[4]: Y11[i]=3
Y11 = np.array(Y11)
Y111 = np.zeros((X1.shape[0], 4))
print Y1.shape
for i in range(X1.shape[0]):
    Y111[i, Y11[i]] = 1
print X1.shape
def nerualNet1(X,Y1,epochno,mod,W1,W2,b1,b2):
	s = X.shape[0] #  samples
	n = X.shape[1] #  features per sample
	d = 30 # nodes in the hidden layer
	c = 4 # three classes to predict 
	alpha = 10e-3
	for epoch in range(epochno):
	    # forward pass
	    A = sigmoid(X.dot(W1) + b1) 
	    B = softmax(A.dot(W2) + b2)
	    loss = cross_entropy_softmax_loss_array(B, Y1)
	    loss += regularization_L2_softmax_loss(0.001, W1, W2)
	#accuracy
	    ans=0
	    for i in range(X.shape[0]):
	    	temp = map(int,np.unravel_index(B[i].argmax(),B[i].shape))
	    	temp1= map(int,np.unravel_index(Y1[i].argmax(),Y1[i].shape))
	    	# print temp1 , temp
	    	if temp!=temp1:
	    		ans+=1
	    # print ans
	    fans = 100 - ans/float(X.shape[0])*100
	    if epoch%mod==0:
	    	print epoch,"===>",fans,"===>",loss

nerualNet1(X1,Y111,1,1,W1,W2,b1,b2)