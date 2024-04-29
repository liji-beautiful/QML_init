from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import time
import numpy as np
#导入数据集
wine = datasets.load_wine() 

#划分数据集
wine_data=wine.data
wine_target=wine.target
wine_data_train, wine_data_test, wine_target_train, wine_target_test = train_test_split(wine_data, wine_target, test_size=0.2, random_state=42)

#标准化处理
stdScale=MinMaxScaler().fit(wine_data)
data_train=stdScale.transform(wine_data_train)
data_test=stdScale.transform(wine_data_test)

feature_num=data_train.shape[1]

target_train=wine_target_train
target_test=wine_target_test

target_test=np.eye(np.max(target_test)+1)[target_test.reshape(-1)]
target_train=np.eye(np.max(target_train)+1)[target_train.reshape(-1)]

def transfom_scale(item,eps,object_base):
    result=''
    precision=math.ceil(math.log(eps**(-1),object_base))
    
    for _ in range(precision):
        item *= object_base
        digit = int(item)
        result += str(digit)
        item -= digit
     
    return result


def conversion_of_number_systems(eps,object_base=4,clifford_group=["I","x","y","z"],data_train=data_train,data_test=data_test):
    assert object_base==len(clifford_group), "the length of clifford must be same to target scale"
    data_train_str=[['' for _ in range(data_train.shape[1])] for _ in range(data_train.shape[0])]
    data_test_str=[['' for _ in range(data_test.shape[1])] for _ in range(data_test.shape[0])]

    for j in range(data_train.shape[0]):
        for i in range(data_train.shape[1]):
            data_train_str[j][i]=transfom_scale(data_train[j][i],eps,object_base)
            
    for j in range(data_test.shape[0]):
        for i in range(data_test.shape[1]):
            data_test_str[j][i]=transfom_scale(data_test[j][i],eps,object_base)
       
    return data_train_str,data_test_str

