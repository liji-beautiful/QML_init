import numpy as np
import math

def softmax(x):
    #print(x)
    exp_x = np.exp(x)
    #print(exp_x)
    #print(exp_x / np.sum(exp_x))
    return exp_x / np.sum(exp_x)

def mse_loss(prediction, target):
    
    loss=(prediction-target)**2
    #print(loss)
    #print(type(loss))
    return loss

def hinge_loss(predictions, targets):
    """Implements the hinge loss."""
    
    hinge_loss = 1 - predictions * targets
    # trick: since the max(0,x) function is not differentiable,
    # use the mathematically equivalent relu instead
    hinge_loss = max(0,hinge_loss)
    return hinge_loss

def generate_observables_list(n,sampled_index=None): 
    if sampled_index==None:
        sampled_index=list(range(0, 4**n))
        
    observables_list=[]
    Paulis = ["X","Y","Z","I"]
    
    for i in sampled_index:
        
        binary_str =decimal_to_quaternary(i).zfill(n)
        xyzistr = [Paulis[int(b)] for k,b in enumerate(binary_str)]
        XYZI=""
        #print(xyzistr)
        for j in range(n):
            XYZI+=xyzistr[j]
        observables_list.append(XYZI)
    return observables_list
            
            
def decimal_to_quaternary(decimal_num):
    if decimal_num == 0:
        return '0'
    
    quaternary_num = ''
    while decimal_num > 0:
        remainder = decimal_num % 4
        quaternary_num = str(remainder)+quaternary_num
        decimal_num = decimal_num // 4
    #print(quaternary_num)
    return quaternary_num

