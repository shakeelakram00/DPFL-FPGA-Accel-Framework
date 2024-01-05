import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")#########TO SUPRESS THE WARNINGS FROM THE OUTPUT ###RESET TO DEFAUL ON LAST LINE OF THE CODE
warnings.filterwarnings("ignore", category=UserWarning, module="OpenMP")
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'   ######TO SUPRESS "OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option."
import copy
import time
import pickle
import numpy as np
import torch
import brevitas.onnx as bo
from brevitas.nn import QuantLinear, QuantReLU, QuantConv2d, QuantMaxPool2d, QuantIdentity, QuantConv1d, QuantMaxPool1d

from torchsummary import summary

from options import args_parser
from update_s4 import LocalUpdate
from utils import test_inference
from utils import average_weights, exp_details
from logging_results import logging

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine

import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix


import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from Models import CNV_i, CNVdp

import random
#from google.colab import files
import io

from tensor_norm import TensorNorm
from common import CommonWeightQuant, CommonActQuant
from brevitas.core.restrict_val import RestrictValueType

from DatasetSplitByUserAndEpoch import extract_datasets, split_user_datasets_by_epochs, datasetSaperateXY, Data
#################################################################################################
from npytodat import make_weight_file
import json
# Load the .json configuration file of FINN for PE and SIMD of layers i.e. matrixvectoractivation
with open('AIAccel4_hw_config.json', 'r') as file:
    config_data = json.load(file)
#################################################################################################
if __name__ == '__main__':
    
    ############# Common ###################
    args = args_parser()
#     
#     #####################################################################################################
#     # # Usage example: Now for each user in the user_xxxxx_dataset; wher xxxxx can xtrain, ytrain, xval, yval
#     user_perEpoch_xtrain_datasets = split_user_datasets_by_epochs(userAAA_xtrain_datasets[0], epochs=args.epochs)#epochs=2)
    ########################################################################################################
    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'    
    
    # load dataset and user groups
    xtrainT, ytrainT, xvalT, yvalT, xtestT, ytestT = datasetSaperateXY()
    global_xtrain_dataset, user_xtrain_datasets = extract_datasets(xtrainT, users=args.num_users)#users=2)
    global_ytrain_dataset, user_ytrain_datasets = extract_datasets(ytrainT, users=args.num_users)#users=2)
    global_xval_dataset, user_xval_datasets = extract_datasets(xvalT, users=args.num_users)#users=2)
    global_yval_dataset, user_yval_datasets = extract_datasets(yvalT, users=args.num_users)#users=2)
    global_xtest_dataset, user_xtest_datasets = extract_datasets(xtestT, users=args.num_users)#users=2)
    global_ytest_dataset, user_ytest_datasets = extract_datasets(ytestT, users=args.num_users)#users=2)
    
    global_train_data = Data(global_xtrain_dataset, global_ytrain_dataset)
    global_val_data = Data(global_xval_dataset, global_yval_dataset)
    global_test_data = Data(global_xtest_dataset, global_ytest_dataset)
    
    #train_dataset, test_dataset, test_data, user_groups = get_dataset(args)
    

    # BUILD MODEL
    if args.model == 'cnn':     
        
        # Convolutional neural netork
        if args.dataset == '5Ectopic':
            #torch.manual_seed(0)  # setting seeds for reproducibility
            
            global_model = CNV_i(num_classes=5, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_ch=1)
            global_model_dp = CNVdp(num_classes=5, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_ch=1)
            global_model.load_state_dict(torch.load('DPFL_AIAccel34_state_dict.pth'))
            global_model_dp.load_state_dict(torch.load('DPFL_AIAccel34_state_dict.pth'))
            
    
            
            def print_summary(model):
            	total_params = 0
            	trainable_params = 0
            	non_trainable_params = 0
            	layer_num = 0
            	print("\n________________________________________________________________________________")
            	print("Layer (type)                           Output Shape                    Param #")
            	print("--------------------------------------------------------------------------------")
            	for name, param in model.named_parameters():
            		if param.requires_grad:
            			trainable_params += param.numel()
            		else:
            			non_trainable_params += param.numel()
            		total_params += param.numel()
            		layer_num += 1
            		layer_name = name.split('.')[0]
            		output_shape = list(param.size())
            		print(f"{layer_num:3d} {layer_name:<34s}{str(output_shape):28s}{param.numel():10d}")
            	print("_______________________________________________________________________________")
            	print("Total parameters =", total_params)
            	print("Trainable parameters =", trainable_params)
            	print("Non-trainable parameters =", non_trainable_params)
            	print("-------------------------------------------------------------------------------")
            	# Rest of the code for memory size estimation remains the same
            	input_size = (1, 14, 14)  # Example input size
            	print("Input Size (MB):", input_size[0] * input_size[1] * input_size[2] * 4 / (1024 * 1024))
            	print("Forward/Backward pass size (MB):", 2 * trainable_params * 4 / (1024 * 1024))
            	print("Param Size (MB):", total_params * 4 / (1024 * 1024))
            	print("Estimated Total Size (MB) :", (input_size[0] * input_size[1] * input_size[2] + 2 * trainable_params + total_params) * 4 / (1024 * 1024))
            	
            #print_summary(global_model)
            

            
            
            
            for i, child in enumerate(global_model_dp.linear_features.children()):
#                 specific_values_of_i = [5, 5, 9]
                if i == 1 or i == 4 or i == 7:             #i == 7:  # Assuming the 6th layer is at index 5 (0-indexed)
                    for param in child.parameters():
                        param.requires_grad = False
            for i, child in enumerate(global_model_dp.conv_features.children()):
#                 specific_values_of_i = [5, 5, 9]
                if i == 2 or i == 5 or i == 9:             #i == 7:  # Assuming the 6th layer is at index 5 (0-indexed)
                    for param in child.parameters():
                        param.requires_grad = False                   
                    
                    
                    
            for i, child in enumerate(global_model.linear_features.children()):
#                 specific_values_of_i = [5, 5, 9]
                if i == 7:  # Assuming the 6th layer is at index 5 (0-indexed)
                    for param in child.parameters():
                        param.requires_grad = False
         

    else:
        exit('Error: unrecognized model')
    ############# Common ###################

    ######### DP Model Compatibility #######
    if args.withDP:
        try:
            inspector = DPModelInspector()
            inspector.validate(global_model_dp)
            print("Model's already Valid!\n")
        except:
            global_model_dp = module_modification.convert_batchnorm_modules(global_model_dp)
            inspector = DPModelInspector()
            print(f"Is the model valid? {inspector.validate(global_model_dp)}")
            print("Model is convereted to be Valid!\n")        
    ######### DP Model Compatibility #######

    
    ######### Local Models and Optimizers #############
    local_models = []
    local_models_dp = []
    local_optimizers = []
    local_privacy_engine = []
    

    for u in range(args.num_users):
        local_models.append(copy.deepcopy(global_model))
        local_models_dp.append(copy.deepcopy(global_model_dp))
        

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_models_dp[u].parameters(), lr=args.lr, #############################
                                        momentum=args.momentum)        
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_models_dp[u].parameters(), lr=args.lr)
            
        if args.withDP:
            # This part is buggy intentionally. It makes privacy engine avoid giving error with vhp.
            epoch_xtrain_datasets = split_user_datasets_by_epochs(user_xtrain_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_ytrain_datasets = split_user_datasets_by_epochs(user_ytrain_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            
            
            user_train_data = Data(epoch_xtrain_datasets[u], epoch_ytrain_datasets[u])
            #print(len(user_train_data), "..........................................................................")
           
            privacy_engine = PrivacyEngine(
                local_models_dp[u],
                batch_size = int((len(user_train_data)/args.epochs)*args.sampling_prob), 
                sample_size = len(user_train_data), 
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier = args.noise_multiplier/np.sqrt(args.num_users),
                max_grad_norm =  args.max_grad_norm,
            )

            privacy_engine.attach(optimizer)            
            local_privacy_engine.append(privacy_engine)

        local_optimizers.append(optimizer)


    if args.optimizer == 'sgd':
        g_optimizer = torch.optim.SGD(global_model_dp.parameters(), lr=args.lr,       ##############################
                                    momentum=args.momentum)        
    elif args.optimizer == 'adam':
        g_optimizer = torch.optim.Adam(global_model_dp.parameters(), lr=args.lr)    
    if args.withDP:
        actual_train_ds_size = len(global_train_data)
        global_privacy_engine = PrivacyEngine(
            global_model_dp,
            batch_size = int(actual_train_ds_size*args.sampling_prob),
            sample_size = actual_train_ds_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier = args.noise_multiplier,
            max_grad_norm =  args.max_grad_norm)  
        global_privacy_engine.attach(g_optimizer)
    ######## Local  Models and Optimizers #############

    # Training
    train_loss = []
    test_log = []
    epsilon_log = []
    
    print("Avg Local batch_size: ", int(len(user_train_data)*args.sampling_prob))
    print("Avg Global batch_size: ", int(actual_train_ds_size*args.sampling_prob))

    for epoch in range(args.epochs):    
        ## Sample the users ##        
        idxs_users = np.random.choice(range(args.num_users),
                                      max(int(args.frac * args.num_users), 1),
                                      replace=False)
        #####
        local_weights, local_losses = [], [] 
        #print("Iterations...............................", epoch, args.epochs, idxs_users)
        

        for u in idxs_users:
            ###empty_cache for cpu is required            
            torch.cuda.empty_cache()
            epoch_xtrain_datasets = split_user_datasets_by_epochs(user_xtrain_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_ytrain_datasets = split_user_datasets_by_epochs(user_ytrain_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_xval_datasets = split_user_datasets_by_epochs(user_xval_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_yval_datasets = split_user_datasets_by_epochs(user_yval_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_xtest_datasets = split_user_datasets_by_epochs(user_xtest_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_ytest_datasets = split_user_datasets_by_epochs(user_ytest_datasets[u], 
                                                                  epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            
            
            user_train_data = Data(epoch_xtrain_datasets[epoch], epoch_ytrain_datasets[epoch])
            user_val_data = Data(epoch_xval_datasets[epoch], epoch_yval_datasets[epoch])
            user_test_data = Data(epoch_xtest_datasets[epoch], epoch_ytest_datasets[epoch])
            #print("Iterations...............................", epoch, args.epochs, idxs_users, u)
            

            local_model = LocalUpdate(args=args, train_dataset=user_train_data,
            			      val_dataset=user_val_data, test_data=user_test_data, # #validation loader has been called test loader because it may have used as testloader in OS-DPFL... test_dataset=val_data
                                      u_id=u, 
                                      sampling_prob=args.sampling_prob,
                                      optimizer = local_optimizers[u])
            #print("Iterations.....localupdate..........................", epoch, args.epochs, idxs_users, u)

            w, loss, local_optimizers[u] = local_model.update_weights(
                                                    model=local_models_dp[u],
                                                    global_round=epoch)
            #print("Iterations.....updateweights..........................", epoch, args.epochs, idxs_users, u)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            print('Global Iteration:',epoch+1, 'Completed for User:', u+1)

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        
        #############################################################################################################
        # Export to ONNX
#         input_shape = (1, 1, 1, 187)
#         input_a = np.random.uniform(0, 1.9999999, size=input_shape).astype(np.float32)
#         input_onnx = torch.from_numpy(input_a)
#         bo.export_finn_onnx(global_model, export_path='Global-DPFL-DNN-5Ectopic.onnx', input_t=input_onnx)
#         print("Global ONNX Model updated to disk")
          ########################################################################################################        
        

        for u in range(args.num_users):
            local_models.append(copy.deepcopy(global_model))###local model for inference only
            local_models[u].load_state_dict(global_weights)##global_weights from dpfl model
            #torch.save(local_models[u].state_dict(), 'DPWL_user_weight.pth')
            epoch_xtest_datasets = split_user_datasets_by_epochs(user_xtest_datasets[u],
                                                                 epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            epoch_ytest_datasets = split_user_datasets_by_epochs(user_ytest_datasets[u],
                                                                 epochs=args.epochs)#epochs=2)user_xtrain_datasets[u]
            user_test_data = Data(epoch_xtest_datasets[epoch], epoch_ytest_datasets[epoch])
            if u == 0:  ###Assume User 1 is Accelerator
#                 print(f"\nUser{u+1} (Accel) Inference With Updated Weights")
#                 _acc, _loss = test_inference(args, local_models[u], user_test_data)
#                 print(f"Accuracy: {round(_acc,5)*100} % \nLoss: {_loss}\n")
                
                #variable_in_path = str(round(_acc,5)*100).replace('.', '_')
                #dat_accV_path = f"./User{u+1}_Acc{variable_in_path}_datWeights"
                
                #os.makedirs(dat_accV_path, exist_ok=True)
                #torch.save(local_models[u].state_dict(), dat_accV_path+'/model_state_dict.pth')
                for i, (name, param) in enumerate(local_models[u].named_parameters()):
                    #print("i", i, param.shape)
                    ###For conversion of generated .npy weights into the format required for generating
                    ###.dat file###############
                    dat_files_path = './pynq_driver_yp_4wj_1/runtime_weights/'
#                     To update the existing files for DPFL and weight updates of accelerator layers. .
                    dat_files_path = './runtime_weights/'
                    if i == 0 or i == 4 or i == 8 or i == 12 or i==16 or i==20:
                        param = param.detach().numpy()
                        np.save(dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', param)
                        
                    if i==0:   #For fist layer we have bipolar weights
                        CNV1npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        CNV1npyBip = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
                        CNV1npyBip[CNV1npyBip < 0] = -1
                        CNV1npyBip[CNV1npyBip > 0] = 1
                        np.save(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy',
                            CNV1npyBip)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy',
#                             CNV1npyBip)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get(f"MatrixVectorActivation_{i}", {}).get("PE", None)
                        simd = config_data.get(f"MatrixVectorActivation_{i}", {}).get("SIMD", None)
                        mw, mh = CNV1npyBip.shape
                        make_weight_file(CNV1npyBip, "decoupled_runtime",
                                       dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)
#                         make_weight_file(CNV1npyBip, "decoupled_runtime",
#                                        dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.dat', 
#                                        mw, mh, pe, simd)
                    if i==4: # for cnv2 and cnv3 we have binary weights
                        print(name)
                        CNV1npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        CNV1npyB = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
                        CNV1npyB[CNV1npyB < 0] = 0
                        CNV1npyB[CNV1npyB > 0] = 1
                        np.save(
                            dat_files_path+'2_0_StreamingDataflowPartition_2_MatrixVectorActivation_0.npy', CNV1npyB)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', CNV1npyB)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get("MatrixVectorActivation_1", {}).get("PE", None)
                        simd = config_data.get("MatrixVectorActivation_1", {}).get("SIMD", None)
                        mw, mh = CNV1npyB.shape
                        make_weight_file(CNV1npyB, "decoupled_runtime",
                                       dat_files_path+'2_0_StreamingDataflowPartition_2_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)

                    if i==8: # for cnv2 and cnv3 we have binary weights
                        print(name)
                        CNV1npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        CNV1npyB = CNV1npy.transpose(2, 3, 1, 0).reshape(-1, 21)
                        CNV1npyB[CNV1npyB < 0] = 0
                        CNV1npyB[CNV1npyB > 0] = 1
                        np.save(
                            dat_files_path+'3_0_StreamingDataflowPartition_3_MatrixVectorActivation_0.npy', CNV1npyB)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', CNV1npyB)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get("MatrixVectorActivation_2", {}).get("PE", None)
                        simd = config_data.get("MatrixVectorActivation_2", {}).get("SIMD", None)
                        mw, mh = CNV1npyB.shape
                        make_weight_file(CNV1npyB, "decoupled_runtime",
                                       dat_files_path+'3_0_StreamingDataflowPartition_3_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)
#                         make_weight_file(CNV1npyB, "decoupled_runtime",
#                                        dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.dat',
#                                        mw, mh, pe, simd)
                        
                    if i == 12:  #for linear layers
                        Lin2npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        Lin2npyB = np.transpose(Lin2npy)    #working on lin2 and lin3
                        Lin2npyB[Lin2npyB < 0] = 0
                        Lin2npyB[Lin2npyB > 0] = 1
                        np.save(
                            dat_files_path+'4_0_StreamingDataflowPartition_4_MatrixVectorActivation_0.npy', Lin2npyB)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', Lin2npyB)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get("MatrixVectorActivation_3", {}).get("PE", None)
                        simd = config_data.get("MatrixVectorActivation_3", {}).get("SIMD", None)
                        mw, mh = Lin2npyB.shape
                        make_weight_file(Lin2npyB, "decoupled_runtime",
                                         dat_files_path+'4_0_StreamingDataflowPartition_4_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)

                    if i==16:  #for linear layers
                        Lin2npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        Lin2npyB = np.transpose(Lin2npy)    #working on lin2 and lin3
                        Lin2npyB[Lin2npyB < 0] = 0
                        Lin2npyB[Lin2npyB > 0] = 1
                        np.save(
                            dat_files_path+'5_0_StreamingDataflowPartition_5_MatrixVectorActivation_0.npy', Lin2npyB)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', Lin2npyB)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get("MatrixVectorActivation_4", {}).get("PE", None)
                        simd = config_data.get("MatrixVectorActivation_4", {}).get("SIMD", None)
                        mw, mh = Lin2npyB.shape
                        make_weight_file(Lin2npyB, "decoupled_runtime",
                                       dat_files_path+'5_0_StreamingDataflowPartition_5_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)
                    if i==20:  #for linear layers
                        Lin2npy = np.load(
                            dat_files_path+f'{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy')
                        Lin2npyB = np.transpose(Lin2npy)    #working on lin2 and lin3
                        Lin2npyB[Lin2npyB < 0] = 0
                        Lin2npyB[Lin2npyB > 0] = 1
                        np.save(
                            dat_files_path+'6_0_StreamingDataflowPartition_6_MatrixVectorActivation_0.npy', Lin2npyB)
#                         np.save(
#                             dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.npy', Lin2npyB)
                        #To make .dat file out of compatilble .npy file
                        # Extract values for "PE" and "SIMD" from MatrixVectorActivation_i
                        pe = config_data.get("MatrixVectorActivation_5", {}).get("PE", None)
                        simd = config_data.get("MatrixVectorActivation_5", {}).get("SIMD", None)
                        mw, mh = Lin2npyB.shape
                        make_weight_file(Lin2npyB, "decoupled_runtime", 
                                       dat_files_path+'6_0_StreamingDataflowPartition_6_MatrixVectorActivation_0.dat',
                                       mw, mh, pe, simd)
#                             make_weight_file(Lin2npyB, "decoupled_runtime",
#                                        dat_accV_path+f'/{i+1}_0_StreamingDataflowPartition_{i+1}_MatrixVectorActivation_0.dat',
#                                        mw, mh, pe, simd)
                print(f"User{u+1} (Accel) .dat Weights updated:\n")
                
                print(f"\nUser{u+1} (Accel) Inference With Updated Weights")
                import subprocess
                #Replace {Pass} with FPGA Board Password
                command = f'echo {Pass} | sudo -S python3.6 validate.py --dataset="5ectopic" --batchsize=4527 --bitfile=AI-Accel4.bit'
                # Use subprocess to run the command
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # Wait for the process to finish and get the output
                stdout, stderr = process.communicate()
                # Print the output
                print("Accel Inference Running:")
                print(stdout)
                print("Accel Inference Error/Warning:", stderr)
                # Check if the process was successful
                if process.returncode == 0:
                    print("Accel executed successfully.")
                else:
                    print(f"Error: Accel execution failed with return code {process.returncode}.")

   
            else:  ###Assume User 1 is Accelerator
                print(f"\nUser{u+1} (Arm) Inference With Updated Weights")
                _acc, _loss = test_inference(args, local_models[u], user_test_data)
                print(f"Test Accuracy: {round(_acc,5)*100} % \nTest Loss: {_loss}\n")
            
#             # Export to ONNX
#             bo.export_finn_onnx(global_model, export_path=f'User{u+1}-DPFL-DNN-5Ectopic.onnx', input_t=input_onnx)
#             print(f"User{u+1} ONNX Model updated to disk")
            

        
        print('Global Iteration:',epoch+1, 'Complete Process Completed')
        #if epoch !=0 and epoch%30==0:   # for result on every 31st epoch
        #if epoch !=0 and epoch%2==0:  #for result on every 3rd epoch
        if epoch >=0:        # for every global epoch (like used for complete data set to check accuracy at each global epoch)
            torch.cuda.empty_cache()          
            loss_avg = sum(local_losses) / len(local_losses)        
            train_loss.append(loss_avg)

            _acc, _loss = test_inference(args, global_model, global_test_data)#replace test_dataset with test_data for testing the model for noise free unexposed data.      
            test_log.append([_acc, _loss])  
          
            if args.withDP:
                global_privacy_engine.steps = epoch+1
                epsilons, _ = global_privacy_engine.get_privacy_spent(args.delta)                                        
                epsilon_log.append([epsilons])
            else:
                epsilon_log = None

            logging(args, epoch, train_loss, test_log, epsilon_log)
            #print(global_privacy_engine.steps)
warnings.resetwarnings()
