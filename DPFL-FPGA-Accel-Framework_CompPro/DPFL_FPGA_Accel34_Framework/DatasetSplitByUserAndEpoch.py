import torch
import numpy as np
from torch.utils.data import Dataset


def datasetSaperateXY():
    xtrain = torch.load("xtrain25000014x14float32.pth")
    ytrain = torch.load("ytrain250000int64.pth")
    xval = torch.load("xval4412014x14float32.pth" )
    yval = torch.load("yval44120int64.pth")
    xtest = torch.load("xtest4527014x14float32.pth")
    ytest = torch.load("ytest45270int64.pth")
    return xtrain, ytrain, xval, yval, xtest, ytest

class Data(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)
        self.y = y
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

def generate_user_indices(users, samples_per_class, samples_per_user, spuPgspu):
    num_classes = 5

    class_indices_global_first = [i * samples_per_class for i in range(num_classes)]
    class_indices_rest = [i * samples_per_class + spuPgspu for i in range(num_classes)]

    global_indices = []
    user_indices = [[] for _ in range(users)]

    # Generate indices for global dataset
    for class_num, start_index in enumerate(class_indices_global_first):
        end_index = start_index + spuPgspu
        global_indices.extend(list(range(start_index, end_index)))

    # Generate indices for each user
    for user in range(users):
        user_start_index = user * samples_per_user

        for class_num, start_index in enumerate(class_indices_rest):
            end_index = start_index + samples_per_user - spuPgspu
            user_indices[user].extend(list(range(start_index + user_start_index, end_index + user_start_index + spuPgspu)))
    
    #print("Global range for dataset:", f"{min(global_indices)}:{max(global_indices) + 1}")
    return global_indices, user_indices

# Assuming xtrain is your original dataset
# Extract datasets for global and each user
def extract_datasets(dataset, users):    
    samples_per_class = len(dataset)//5
    spuPgspu = round(samples_per_class*.2)
    samples_per_user = (samples_per_class-spuPgspu) // users   
    global_indices, user_indices_list = generate_user_indices(users, samples_per_class, samples_per_user, spuPgspu)
    global_dataset = dataset[global_indices]

    user_datasets = []
    for user_indices in user_indices_list:
        user_datasets.append(dataset[user_indices])
#     print("Global dataset and shape:", global_dataset, global_dataset.shape)
    #print("Global dataset shape:", global_dataset.shape)
    
    for i in range(users):
        user_range = f"{min(user_indices_list[i])}:{max(user_indices_list[i]) + 1}"
        #print(user_datasets[i])
        #print(f"For User {i + 1} Range: {user_range}")
        #print(f"For User {i + 1} - Shape: {user_datasets[i].shape}")
        #print("\n")
    return global_dataset, user_datasets

def generate_epoch_indices(epochs, E_samples_per_class, E_samples_per_user):
    num_classes = 5
    epoch_indices_rest = [i * E_samples_per_class for i in range(num_classes)]
    user_epoch_indices = [[] for _ in range(epochs)]
    # Generate indices for each epoch
    for epoch in range(epochs):
        epoch_start_index = epoch * E_samples_per_user

        for class_num, start_index in enumerate(epoch_indices_rest):
            end_index = start_index + E_samples_per_user
            user_epoch_indices[epoch].extend(list(range(start_index + epoch_start_index, end_index + epoch_start_index)))
    return user_epoch_indices
    
def split_user_datasets_by_epochs(user_dataset, epochs):
    E_samples_per_class = len(user_dataset)//5
    E_samples_per_user = (E_samples_per_class) // epochs 
    user_epoch_indices_list = generate_epoch_indices(epochs, E_samples_per_class, E_samples_per_user)
    user_epoch_datasets = []
    for epoch_indices in user_epoch_indices_list:
        user_epoch_datasets.append(user_dataset[epoch_indices])
    for i in range(epochs):
        user_epoch_range = f"{min(user_epoch_indices_list[i])}:{max(user_epoch_indices_list[i]) + 1}"
        #print(user_epoch_datasets[i])
        #print(f"For User Epoch{i + 1} Range: {user_epoch_range}")
        #print(f"For User Epoch{i + 1} - Shape: {user_epoch_datasets[i].shape}")
        #print("\n")
    return user_epoch_datasets


####################SampleRuns#######################################################################
# global_xtrain_dataset, user_xtrain_datasets = extract_datasets(xtrain, users=args.num_users)#users=2)
# global_ytrain_dataset, user_ytrain_datasets = extract_datasets(ytrain, users=args.num_users)
# global_xval_dataset, user_xval_datasets = extract_datasets(xval, users=args.num_users)
# global_yvaldataset, user_yval_datasets = extract_datasets(yval, users=args.num_users)
# global_xtest_dataset, user_xtest_datasets = extract_datasets(xtest, users=args.num_users) #user_xtest_datasets[0/1] for user 0/1
# global_ytest_dataset, user_ytest_datasets = extract_datasets(ytest, users=args.num_users)#user_ytest_datasets[0/1] for user 0/1

#####################################################################################################
# # Usage example: Now for each user in the user_xxxxx_dataset; wher xxxxx can xtrain, ytrain, xval, yval
# user_perEpoch_xtrain_datasets = split_user_datasets_by_epochs(user_xtrain_datasets[0], epochs=args.epochs)#epochs=2)
# user_perEpoch_ytrain_datasets = split_user_datasets_by_epochs(user_ytrain_datasets[0], epochs=args.epochs)
# user_perEpoch_xval_datasets = split_user_datasets_by_epochs(user_xval_datasets[0], epochs=args.epochs)
# user_perEpoch_yval_datasets = split_user_datasets_by_epochs(user_yval_datasets[0], epochs=args.epochs)
