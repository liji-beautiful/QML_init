from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.algorithms import VQC
from dataprocess import target_train,target_test,conversion_of_number_systems,feature_num
import math
import stim
import json
import hypermapper
import sys
import logging
import random
import numpy as np
from circuit_manipulation import qiskit_to_stim,transform_to_allowed_gates
from train_func import *

def ansatz_circuit(qubit_num,ansatz_reps=5):
    ansatz=EfficientSU2(qubit_num, su2_gates=['ry','rz'], parameter_prefix='x',entanglement='circular', reps=ansatz_reps)
    params_num= ansatz.num_parameters
    #print(ansatz.parameters)
    return ansatz,params_num

def encode_circuit(input_params,qubit_num,clifford_group=["I","x","y","z"]):
    #print(input_params)
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

def vqc_circuit(input_params,guess_params,qubit_num=feature_num,ansatz_reps=5):
    guess_list=[]
    for i in range(len(guess_params)):
        guess_list.append(guess_params['x'+str(i)])
    assert feature_num>=qubit_num , "特征数量过多无法编码"
    
    encoded_circuit=encode_circuit(input_params,qubit_num)
    
    ansatz,_ =ansatz_circuit(qubit_num)
    
    ansatz=ansatz.assign_parameters(guess_list)
    
    ansatz=ansatz.decompose()
    
    vqc_cir=QuantumCircuit(qubit_num)
    vqc_cir.compose(encoded_circuit,inplace=True)
    vqc_cir.compose(ansatz,inplace=True)
    
    return vqc_cir

def single_data_output(input_params,target,guess_param,qubit_num):
    vqc=vqc_circuit(input_params,guess_param)

    vqc_clif=transform_to_allowed_gates(vqc)
    
    vqc_stim=qiskit_to_stim(vqc_clif)
    
    sim = stim.TableauSimulator()
    sim.do_circuit(vqc_stim)
    observable_1="Z"+"I"*(qubit_num-1)
    observable_2="I"+"Z"+"I"*(qubit_num-2)
    observable_3="II"+"Z"+"I"*(qubit_num-3)
    expection_val_1=sim.peek_observable_expectation(stim.PauliString(observable_1))
    expection_val_2=sim.peek_observable_expectation(stim.PauliString(observable_2))
    expection_val_3=sim.peek_observable_expectation(stim.PauliString(observable_3))
    print(expection_val_1,expection_val_2,expection_val_3)
    expection_classifier=[expection_val_1,expection_val_2,expection_val_3]
    prob_distri=softmax(expection_classifier)
    single_loss=cross_entropy_loss(prob_distri,target)
    return single_loss

def vqc_stim(guess_param,data_train,qubit_num=feature_num):
    loss=0
    #print(data_train)
    for i in range(len(data_train)):
        loss+=single_data_output(data_train[i],target_train[i],guess_param,qubit_num)
    return loss/len(data_train)


def vqc_init_run(itera,qubit_num=feature_num,eps=4**(-3),object_base=4,clifford_group=["I","x","y","z"]):
    data_train,_=conversion_of_number_systems(eps,object_base,clifford_group)
    _,params_num=ansatz_circuit(qubit_num)
    hypermapper_config_path = "D:\\QML\\QML_Cliiford\\hypermapper_config.json"
    config = {}
    config["application_name"] = "vqc_init"
    config["optimization_objectives"] = ["value"]
    config["optimization_iterations"] = itera
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(params_num):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, math.pi/2, math.pi, -math.pi/2]
        x["parameter_default"] = math.pi/2  #random.randint(-1, 2)*math.pi/2
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] ='D:\\QML\\QML_Cliiford\\hypermapper_log.log'
    config["output_data_file"] = "D:\\QML\\QML_Cliiford\\hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    hypermapper.optimizer.optimize(
        hypermapper_config_path, 
        lambda x: vqc_stim(
            guess_param=x,
            data_train=data_train
                           )
    )

#vqc_init_run(10)