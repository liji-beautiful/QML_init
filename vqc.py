from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.algorithms import VQC
from dataprocess import conversion_of_number_systems,self_make_data
import math
import stim
import csv
import json
import hypermapper
import sys
import logging
import random
import numpy as np
from circuit_manipulation import qiskit_to_stim,transform_to_allowed_gates
from train_func import *
from timeit import default_timer as timer
from itertools import product

def generate_strings(n):
    # 使用itertools.product生成所有长度为n的I和Z的组合
    combinations = product('IZ', repeat=n)
    # 将生成的组合转换为字符串并存储在列表中
    result = [''.join(combination) for combination in combinations]
    return result


def ansatz_circuit(qubit_num,ansatz_reps=4):
    ansatz=EfficientSU2(qubit_num, su2_gates=['ry','rz'], parameter_prefix='x',entanglement='linear', reps=ansatz_reps)
    params_num= ansatz.num_parameters
    #print(ansatz.parameters)
    return ansatz,params_num

def encode_circuit(input_params,qubit_num,clifford_group=["I","x","y","z"]):
    print(input_params)
    encode_qc=QuantumCircuit(qubit_num)
    for i,item in enumerate(input_params):
        #后续还是要考虑一下cliiford更大的时候怎么搞
        for gate_index in item:
            if gate_index=="1":
                encode_qc.x(i)
            elif gate_index=="2":
                encode_qc.y(i)
            elif gate_index=="3":
                encode_qc.z(i)
    return encode_qc

def encode_RealAmplitudes(input_params,qubit_num):
    for i,item in enumerate(input_params):
        
        input_params[i]=math.floor(item*4)-1
        if(input_params[i]>=3):
            input_params[i]=2  
        input_params[i]*=math.pi*0.5
    input_params1 = np.append(input_params, input_params)
    #input_params2 = np.append(input_params1, input_params1)
    encode_qc=RealAmplitudes(qubit_num,reps=1,entanglement='linear')
    #print(encode_qc.num_parameters)
    #print(len(input_params2))
    encode_qc = encode_qc.assign_parameters(input_params1).decompose()
    return encode_qc
    

def vqc_circuit(input_params,guess_params,qubit_num,ansatz_reps=5):
    guess_list=[]
    if type(guess_params)==dict:   
        for i in range(len(guess_params)):
            guess_list.append(guess_params['x'+str(i)])
    else:
        guess_list=guess_params
    #encoded_circuit=encode_RealAmplitudes(input_params,qubit_num)

    ansatz,num_ =ansatz_circuit(qubit_num)
    
    ansatz=ansatz.assign_parameters(guess_list)
   
    ansatz=ansatz.decompose()
    
    vqc_cir=QuantumCircuit(qubit_num)
    #vqc_cir.compose(encoded_circuit,inplace=True)
    vqc_cir.compose(ansatz,inplace=True)
    return vqc_cir

def single_data_output(input_params,target,guess_param,qubit_num,ansatz_reps,observable,params):
    vqc=vqc_circuit(input_params,guess_param,qubit_num,ansatz_reps)
    vqc_clif=transform_to_allowed_gates(vqc)
    vqc_stim=qiskit_to_stim(vqc_clif)
    
    sim = stim.TableauSimulator()
    sim.do_circuit(vqc_stim)


    expect_value=0
    for i in range(len(observable)):
        expect_value+=params[i]*sim.peek_observable_expectation(stim.PauliString(observable[i]))

    print(expect_value,target)
    #with open("D:\\QML\\QML_Cliiford\\samples_no_coeff.csv","a") as file:
    #    writer = csv.writer(file)
    #    writer.writerow([expect_value])
       
    #single_loss=hinge_loss(expect_value,target)
    #print(single_loss)
    return expect_value

def vqc_stim(guess_param,data_train,target_train,qubit_num,ansatz_reps,observable,params,loss_file=None):
 
    loss=0
    start=timer()
   
    for i in range(len(data_train)):
        
        x=np.copy(data_train[i])
        loss+=single_data_output(x,target_train[i],guess_param,qubit_num,ansatz_reps,observable,params)

    end=timer()
    
    if loss_file is not None:
        with open(loss_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss/len(data_train),end - start])
    
    return loss/len(data_train)


def vqc_init_run(itera,qubit_num=0,eps=4**(-3),object_base=4,clifford_group=["I","x","y","z"]):
    data_train_,target_train_,feature_num,data_test,target_test=self_make_data()
    if qubit_num==0:
        qubit_num=feature_num
    #data_train,_=conversion_of_number_systems(eps,object_base,clifford_group)
    #print(qubit_num)
    _,params_num=ansatz_circuit(qubit_num)
    hypermapper_config_path = "D:\\QML\\QML_Cliiford\\50qubits\\hypermapper_config.json"
    config = {}
    config["application_name"] = "vqc_init"
    config["optimization_objectives"] = ["value"]
    config["optimization_iterations"] = itera
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = False
    config["number_of_cpus"]= 1
    config["print_posterior_best"] = False
    config["design_of_experiment"]={}
    config["design_of_experiment"]["doe_type"]="random sampling"
    config["design_of_experiment"]["number_of_samples"]=500
    for i in range(params_num):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, 1.570796327, 3.141592654, -1.570796327]
        x["parameter_default"] = random.choice([0, 1.570796327, 3.141592654, -1.570796327])
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] ='D:\\QML\\QML_Cliiford\\50qubits\\hypermapper_log.log'
    config["output_data_file"] = "D:\\QML\\QML_Cliiford\\50qubits\\hypermapper_output.csv"
    config["resume_optimization"]=True
    config["resume_optimization_data"]="D:\\QML\\QML_Cliiford\\50qubits\\resume_0.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    loss_file='D:\\QML\\QML_Cliiford\\50qubits\\loss_file.csv'
    hypermapper.optimizer.optimize(
        hypermapper_config_path, 
        lambda x: vqc_stim(
            guess_param=x,
            data_train=data_train_,
            target_train=target_train_,
            qubit_num=qubit_num,
            loss_file=loss_file
                           )
    )

if __name__=="__name__":
    begin=timer()
    vqc_init_run(10,50)
    end=timer()
    print(end-begin)


























