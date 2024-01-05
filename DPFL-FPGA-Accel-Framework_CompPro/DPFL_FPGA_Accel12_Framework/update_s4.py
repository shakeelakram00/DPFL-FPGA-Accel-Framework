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
        
        #train_dataset, test_dataset, test_data, user_groups = get_dataset(args)
        args = args_parser()
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.local_bs, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.local_bs, shuffle=True) #validation loader has been called test loader because it may have used as testloader in OS-DPFL... test_dataset=val_data
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.local_bs, shuffle=True)
        
        return train_dataloader, val_dataloader, test_dataloader
        ###########

        
    def update_weights(self, model, global_round): #model
        #Set mode to train model
        model.to(self.device)
        model.train()
        epoch_loss = []
        
#         training_loss_history =[]
        
#         trainer = ptt.create_default_trainer(model, self.optimizer, self.criterion, verbose=1)
#         trainer.register_post_iteration_callback(ptt.callback.ValidationCallback(self.valloader,
#                                                     metric=ptt.metric.TorchLoss(self.criterion)), frequency=200)
#         validation_callback = ptt.callback.ValidationCallback(self.valloader,metric=ptt.metric.TorchLoss(self.criterion))
#         trainer.register_post_epoch_callback(validation_callback, frequency=1)
        
#         # Create a callback to manually log training loss
#         class LogTrainingLossCallback(ptt.Callback):
#             def __call__(self, trainer):
#                 training_loss = trainer.callback_manager.get_metrics('train_loss')
#                 training_loss_history.append(training_loss)
#                 # Register the training loss callback
        
#         trainer.register_post_iteration_callback(LogTrainingLossCallback(), frequency=1)
        
#         accuracy_callback = ptt.callback.MetricCallback(metric=ptt.metric.Accuracy(
#             prediction_transform=lambda x: x.argmax(dim=1, keepdim=False)))
#         trainer.register_post_iteration_callback(accuracy_callback, frequency=1)
#         trainer.add_progressbar_metric("validation loss %.4f | accuracy %.4f", [validation_callback, accuracy_callback])
#         trainer.train(self.trainloader, max_epochs=self.args.local_ep)
        
#         # Access the training and validation loss history using the logger
#         # Manually retrieve training and validation losses
#         validation_loss_history = trainer.callback_manager.get_metrics('val_loss')
        
#         print('Local Epoch:',iter+1, 'Completed')
        
#         return model.state_dict(), sum(training_loss_history) / len(training_loss_history), self.optimizer
        
        for iter in range(self.args.local_ep):            
            batch_loss = []
            self.optimizer.zero_grad()
            #print("..............................", self.args.local_ep)
            if self.args.withDP:
                virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)  
            
            train_correct = 0
            train_total = 0 
            train_batch_loss =[]
            
            total_batches = len(self.trainloader)
#             print(total_batches, "......................................................lkih")
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
                    if i == 5:  # Assuming the third layer is at index 2 (0-indexed)
                        for param in child.parameters():
                            param.requires_grad = False
#                 with torch.no_grad():
#                     for param in model.conv_features.parameters():
#                         param.requires_grad = False
# #                     for param in global_model.conv2.parameters():
# #                         param.requires_grad = False
# #                     for param in global_model.conv3.parameters():
# #                         param.requires_grad = False
#                     for param in model.linear_features.parameters():
#                         param.requires_grad = False
# #                     for param in model.fc2.parameters():
# #                         param.requires_grad = False
# #                     for param in model.fc3.parameters():
# #                         param.requires_grad = False
# #                     for param in model.TensorNorm1.parameters():
# #                         param.requires_grad = False 
               
                model_preds = model(xtrain)
                _, pred_labels = torch.max(model_preds, 1)
                train_correct += torch.sum(pred_labels == ytrain).item()
                train_total += ytrain.size(0)
                loss = self.criterion(model_preds, ytrain)
                train_batch_loss.append(loss.item())
                loss.backward()
                if self.args.withDP:
                    # take a real optimizer step after N_VIRTUAL_STEP steps t                                        
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()                        
                    else:                        
                        self.optimizer.virtual_step() # take a virtual step                        
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                #############
                batch_loss.append(loss.item())
                
                progress_bar.update(1)
                progress_bar.set_postfix(batch=batch_idx + 1, refresh=True)
                
            progress_bar.close()    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            #print('Local Epoch:',iter+1, 'Completed')
            model.eval()  # Set the model to evaluation mode
            val_correct = 0
            val_total = 0
            val_batch_loss = []
            with torch.no_grad():
                for batch_idx, (xval, yval) in enumerate(self.valloader):
                    xval, yval = xval.to(self.device), yval.to(self.device)
                    val_model_preds = model(xval)
                    val_loss = self.criterion(val_model_preds, yval)
                    val_batch_loss.append(val_loss.item())
                    # Calculate validation accuracy
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
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.optimizer
        
#        trainer = ptt.create_default_trainer(model, self.optimizer, self.criterion, verbose=1)
#        trainer.register_post_iteration_callback(ptt.callback.ValidationCallback(self.testloader,
#        metric=ptt.metric.TorchLoss(self.criterion)), frequency=200)
#        validation_callback = ptt.callback.ValidationCallback(self.testloader, metric=ptt.metric.TorchLoss(self.criterion))
#        trainer.register_post_epoch_callback(validation_callback, frequency=1)
#        accuracy_callback = ptt.callback.MetricCallback(metric=ptt.metric.Accuracy(prediction_transform=lambda x: x.argmax(dim=1, keepdim=False)))
#       trainer.register_post_iteration_callback(accuracy_callback, frequency=1)
#        trainer.add_progressbar_metric("validation loss %.4f | accuracy %.4f", [validation_callback, accuracy_callback])
#        trainer.train(self.trainloader, max_epochs=self.args.local_ep)
#        return























