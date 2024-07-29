from sklearn import datasets  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import time
import numpy as np
from sklearn.datasets import make_classification


def wine_data():
    wine=datasets.load_wine()
    selected_indices = np.where((wine.target == 0) | (wine.target == 1))[0]
    wine_data = wine.data[selected_indices]
    wine_target = wine.target[selected_indices]
    wine_target[wine_target == 0] = -1
    wine_data_train, wine_data_test, wine_target_train, wine_target_test = train_test_split(wine_data, wine_target, test_size=0.2, random_state=42)
    #print(wine_data_train)
    #print(wine_target_train)
    #标准化处理
    stdScale=MinMaxScaler().fit(wine_data)
    data_train=stdScale.transform(wine_data_train)
    data_test=stdScale.transform(wine_data_test)

    feature_num=data_train.shape[1]
    
    target_train=wine_target_train
    target_test=wine_target_test
    return data_train,target_train,feature_num,data_test,target_test

def data_iris():
    wine = load_iris() 
    selected_indices = np.where((wine.target == 0) | (wine.target == 1))[0]
    iris_data = iris.data[selected_indices]
    iris_target = iris.target[selected_indices]
    iris_target[iris_target == 0] = -1
    iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)

    #标准化处理
    stdScale=MinMaxScaler().fit(iris_data)
    data_train=stdScale.transform(iris_data_train)
    data_test=stdScale.transform(iris_data_test)

    feature_num=data_train.shape[1]

    target_train=iris_target_train
    target_test=iris_target_test
    return data_train,target_train,feature_num,data_test,target_test

def self_make_data(n_samples=100,n_features=100,n_classes=2,n_train=80):
    #print("#######")
    X, y = make_classification(
        n_samples=n_samples,     # 样本数量
        n_features=n_features,      # 特征数量
        n_informative=2,   # 有用特征数量
        n_redundant=0,     # 冗余特征数量
        n_clusters_per_class=1,
        n_classes=n_classes,       # 分类数量
        random_state=42    # 随机种子
    )
    y[y == 0] = -1
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X[:n_train],y[:n_train],n_features,X[n_train:],y[n_train:]

def transfom_scale(item,eps,object_base):
    
    result=''
    precision=math.ceil(math.log(eps**(-1),object_base))
    
    for _ in range(precision):
        item *= object_base
        digit = int(item)
        result += str(digit)
        item -= digit
    #print(result)
    return result




def conversion_of_number_systems(eps,object_base=4,clifford_group=["I","x","y","z"],data_train=[],data_test=[]):
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

