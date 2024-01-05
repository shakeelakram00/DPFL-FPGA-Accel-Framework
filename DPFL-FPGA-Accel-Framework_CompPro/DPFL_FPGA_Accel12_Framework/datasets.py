import numpy as np
import copy
import torch
import matplotlib.image as mpimg
import urllib.request
import zipfile
import os
import pandas as pd
#from torchvision import datasets, transforms
from sampling import dist_datasets_iid, dist_datasets_noniid
from options import args_parser
from torch.utils.data import Dataset, TensorDataset
from sklearn.utils import resample

def get_dataset(args):

    if args.dataset == '5Ectopic':

        if args.dr_from_np == 1:
            xtrain = torch.load("xtrain2500001414reshapeduint8s3.pth")
            ytrain = torch.load("ytrain250000reshapedint64s3.pth")
            xval = torch.load("xval441201414reshapeduint8s3.pth" )
            yval = torch.load("yval44120reshapedint64s3.pth")
            xtest = torch.load("xtest452701414reshapeduint8s3.pth")
            ytest = torch.load("ytest45270reshapedint64s3.pth")
            # shape of  data
            print('Training Dataset Shapes', xtrain.shape, ytrain.shape,'Validation Dataset Shapes', xval.shape, yval.shape, 'Test Dataset Shapes', xtest.shape, ytest.shape)
            
            class Data(Dataset):
            	def __init__(self, X, y):
            		self.X = X.unsqueeze(1)
            		self.y = y
            		self.len = self.X.shape[0]
            	def __getitem__(self, index):
            		return self.X[index], self.y[index]
            	def __len__(self):
            		return self.len
            # Instantiate training and test data
            train_dataset = Data(xtrain, ytrain)
            #val_data = Data(xval, yval)
            test_dataset = Data(xval, yval)   #val_dataset is called test_dataset just for temporary purpose because test_dataset variable is used in other files from Open Source DPFL
            test_data = Data(xtest, ytest)



    if args.sub_dataset_size > 0:
        rnd_indices = np.random.RandomState(seed=0).permutation(len(train_dataset.data))        
        train_dataset.data = train_dataset.data[rnd_indices]
        if torch.is_tensor(train_dataset.targets):
            train_dataset.targets = train_dataset.targets[rnd_indices]    
        else:
            train_dataset.targets = torch.tensor(train_dataset.targets)[rnd_indices]
        train_dataset.data = train_dataset.data[:args.sub_dataset_size]
        train_dataset.targets = train_dataset.targets[:args.sub_dataset_size]
        print("\nThe chosen sub dataset has the following shape:")
        print(train_dataset.data.shape, train_dataset.targets.shape,"\n")        

    if args.iid:                   
        user_groups = dist_datasets_iid(train_dataset, args.num_users)         
    else:
        user_groups = dist_datasets_noniid(train_dataset, args.num_users,
                                            num_shards=1000,                                                
                                            unequal=args.unequal)    
    
    return train_dataset, test_dataset, test_data, user_groups

## For test
#if __name__ == '__main__':
#    args = args_parser()
#    train_dataset, test_dataset, user_groups = get_dataset(args)
