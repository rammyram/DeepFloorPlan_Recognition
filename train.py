from numpy.core.defchararray import index
from numpy.core.fromnumeric import shape
import torch
from torch._C import device, dtype
from torch.nn.modules import loss
import torchvision
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import configuration
from dataloader import FloorPlanDataset
from unet import UNet
from loss import CrossEntropyLoss
from PIL import Image


def evaluate_model(model, dataloader):
    test_scores = []
    model.eval()
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model.forward(inputs)
        _,preds = torch.max(outputs,1)
        targets_mask = targets >= 0
        test_scores.append(np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy()))

        return np.mean(test_scores)



def wandb_initializer(args):
    with wandb.init(project="Deepfloorplan_Recognition",config=args):
        config = wandb.config

        model,train_loader,val_loader,loss_func,optimizer = nn_model(config)
        train(model,train_loader,val_loader,loss_func,optimizer, config)
    return model

def nn_model(config):
    #data_transformers = transforms.Compose([transforms.ToTensor()])

    train_set = FloorPlanDataset(image_dir=configuration.train_data_config.training_set_dir,gt_dir=configuration.train_data_config.train_ground_truth_dir)
    val_set = FloorPlanDataset(image_dir=configuration.validation_data_config.validation_set_dir,gt_dir=configuration.validation_data_config.validation_ground_truth_dir)

    #Loading train and val set
    train_set_loader = DataLoader(train_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)
    val_set_loader = DataLoader(val_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)

    #Build the model
    net = UNet(n_classes=2)

    if configuration.training_config.device.type == 'cuda':
        net.cuda()

    
    #loss_function = torch.nn.BCEWithLogitsLoss()
    #loss_function = torch.nn.BCELoss()
    loss_function = torch.nn.CrossEntropyLoss()
    
    
    optimizer = torch.optim.Adam(net.parameters(),lr=config.lr)

    return net,train_set_loader,val_set_loader,loss_function,optimizer


def validation(nn_model,val_set_loader,loss_func,epoch,config):
    print("Validating.....")
    
    nn_model.eval()

    val_loss = 0.0
    mini_batches = 0

    for batch_id,(image,gt) in enumerate(val_set_loader):
        if(configuration.training_config.device.type == 'cuda'):
            image,gt = image.cuda(), gt.cuda()
        else:
            image,gt = image, gt

        output = nn_model(image)
        loss = loss_func(output, gt)
       

        mini_batches += 1
        val_loss += float(loss)

        val_loss = val_loss/mini_batches
        print("Validation loss: ",val_loss)
        
        return val_loss


def train(nn_model,train_set_loader,val_set_loader,loss_func,optimizer, config):
    wandb.watch(nn_model,loss_func,log='all',log_freq=10)

    mini_batches = 0
    train_loss = 0.0
    print("Training....")
    for epoch in range(config.epochs):
        #scheduler.step()
        #print("\nLearning rate at this epoch is: %0.9f"%scheduler.get_lr()[0])
        for batch_id,(image,gt) in enumerate(train_set_loader):
            nn_model.train()
            if(configuration.training_config.device.type == 'cuda'):
                image,gt = image.cuda(),gt.cuda()
            else:
                image,gt = image, gt
        
            output = nn_model(image)
            loss = loss_func(output, gt)    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mini_batches += 1
            train_loss += float(loss)

            print("Epoch: " + str(epoch) + " : Mini Batch: " + str(mini_batches) + " Training loss: " + str(train_loss))

            #Plotting in wandb
            if(mini_batches % configuration.training_config.plot_frequency == 0):
                val_loss = validation(nn_model,val_set_loader,loss_func,epoch,config)
                training_log(val_loss,mini_batches)
                training_log(train_loss/mini_batches,mini_batches,False)

                PATH = "model.pt"
                torch.save({'epoch':epoch,'model_state_dict':nn_model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':train_loss},PATH)
                
                train_loss = 0.0
                
            


            print('Epoch-{0} lr:{1:f}'.format(epoch,optimizer.param_groups[0]['lr']))
            print("Score:", evaluate_model(nn_model, val_set_loader))

def training_log(loss,mini_batch,train=True):
    if train == True:
        wandb.log({'batch':mini_batch,'loss':loss})
    else:
        wandb.log({'batch':mini_batch,'loss':loss})


def visualizer(pred_image,epoch,config):
    if (epoch == config.epochs - 1):
        #class_labels = ['void','windows']
        #class_labels_list = np.array([[0],[1]])

        for i in range(2):
            predicted_img = pred_image[i]
            predicted_img[predicted_img > 0.5] = 1.0
            predicted_img[predicted_img < 0.5] = 0.0

        #print(pred_image.shape)
        return predicted_img[0].cpu().detach().cpu()