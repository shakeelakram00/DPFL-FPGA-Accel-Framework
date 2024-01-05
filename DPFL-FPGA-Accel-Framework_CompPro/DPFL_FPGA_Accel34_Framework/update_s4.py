import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from opacus import PrivacyEngine
from datasets import get_dataset
from options import args_parser
import pytorchtrainer as ptt
from tqdm import tqdm


class LocalUpdate(object):
    def __init__(self, args, train_dataset, val_dataset, test_data, u_id, sampling_prob, optimizer):
        self.u_id = u_id
        self.args = args
        self.trainloader, self.valloader, self.testloader = self.train_val_test(
            train_dataset, val_dataset, test_data)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.dataset_size = len(train_dataset) ####!!
        self.sasampling_prob = sampling_prob ####!!
        self.optimizer = optimizer

    def train_val_test(self, train_dataset, val_dataset, test_dataset):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        args = args_parser()
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.local_bs, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.local_bs, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.local_bs, shuffle=True)
        
        return train_dataloader, val_dataloader, test_dataloader
        
        
    def update_weights(self, model, global_round): #model
        #Set mode to train model
        model.to(self.device)
        model.train()
        epoch_loss = []
        batch_loss = []
        
        
        for iter in range(self.args.local_ep):
            model.train()
            self.criterion.train()
            train_correct = 0
            train_total = 0
            train_batch_loss =[]
            if self.args.withDP:
                virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs) 
            
            total_batches = len(self.trainloader)
            progress_bar = tqdm(total=total_batches, desc="Processing Local Epoch(s)", unit="batch", position=0, leave=True)
            
            for batch_idx, (xtrain, ytrain) in enumerate(self.trainloader):
                ####!!
                indices = np.random.permutation(len(ytrain))
                
                rnd_sampled = np.random.binomial(len(ytrain), self.sasampling_prob)
                
                if rnd_sampled > 0:
                    xtrain = xtrain[indices][:rnd_sampled]
                    ytrain = ytrain[indices][:rnd_sampled]
                else:
                    #continue
                    return model.state_dict(), 0., self.optimizer
                ####!!
                
                xtrain, ytrain = xtrain.to(self.device), ytrain.to(self.device)
                for i, child in enumerate(model.linear_features.children()):
                    if i == 7:  # Assuming the third layer is at index 2 (0-indexed)
                        for param in child.parameters():
                            param.requires_grad = False
                
                model_preds = model(xtrain)
                _, pred_labels = torch.max(model_preds, 1)
                train_correct += torch.sum(pred_labels == ytrain).item()
                train_total += ytrain.size(0)
                loss = self.criterion(model_preds, ytrain)
                train_batch_loss.append(loss.item())


                if self.args.withDP:
                    # take a real optimizer step after N_VIRTUAL_STEP steps t                                        
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:                        
                        self.optimizer.zero_grad()
                        loss.backward()# take a virtual step
                        self.optimizer.virtual_step()
                                                
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                #############   
                
                model.clip_weights(-1, 1)
                batch_loss.append(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix(batch=batch_idx + 1, refresh=True)
            progress_bar.close()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            model.eval()  # Set the model to evaluation mode
            self.criterion.eval()
            val_correct = 0
            val_total = 0
            val_batch_loss = []
            
            with torch.no_grad():
                for batch_idx, (xval, yval) in enumerate(self.valloader):
                    xval, yval = xval.to(self.device), yval.to(self.device)
                    val_model_preds = model(xval)
                    val_loss = self.criterion(val_model_preds, yval)
                    val_batch_loss.append(val_loss.item())
                    _, val_pred_labels = torch.max(val_model_preds, 1)
                    val_correct += torch.sum(val_pred_labels == yval).item()
                    val_total += yval.size(0)
                    
            # Calculate and print average training and validation losses and accuracies
            avg_train_loss = sum(train_batch_loss) / len(train_batch_loss)
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0
            avg_val_loss = sum(val_batch_loss) / len(val_batch_loss)
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            print('Local Epoch: ', iter + 1,
                  'Training Loss: ', avg_train_loss, 'Training Accuracy:{:.4f}%'.format(100*train_accuracy))
            print('Validation Loss: ', avg_val_loss, 'Validation Accuracy:{:.4f}%'.format(100*val_accuracy))
            model.train() 
            import csv
            with open('DPFLtraining_log.csv', 'a') as log_file:
                log_file.write(f"{iter+1},{100*train_accuracy},{100*val_accuracy},{avg_train_loss}, {avg_val_loss}\n")
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.optimizer
        























