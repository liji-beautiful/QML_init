import numpy as np
import math

def softmax(x):
    print(x)
    exp_x = np.exp(x)
    print(exp_x)
    print(exp_x / np.sum(exp_x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(prediction, target):
    loss=0
    for i in range(len(target)):
        loss-=target[i]*math.log10(prediction[i])
    print(loss)
    print(type(loss))
    return loss
