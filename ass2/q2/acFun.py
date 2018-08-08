import math
def sigmoid(x):
    try:
        return 1 / (1+math.exp(-x))
    except OverflowError:
        return 0
def tanh(x):
    return math.tanh(x)

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def dtanh(x):
    return (1-tanh(x)*tanh(x))
def softmax(vec):
        ans = 0.0
        for i in vec: ans += pow(math.e, i)
        for i in xrange(len(vec)): vec[i] = pow(math.e, vec[i])/ans
        return vec
