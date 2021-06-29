import torch
from torch._C import device, dtype
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
#from pytorch_toolbelt import losses as L

def wandb_initializer(args):
    with wandb.init(project="Deepfloorplan_Recognition",config=args):
        config = wandb.config

        model,train_loader,val_loader,loss_func,optimizer = nn_model(config)
        train(model,train_loader,val_loader,loss_func,optimizer, config)
    return model

def nn_model(config):
    data_transformers = transforms.Compose([transforms.ToTensor()])

    train_set = FloorPlanDataset(image_dir=configuration.train_data_config.training_set_dir,gt_dir=configuration.train_data_config.train_ground_truth_dir,transform=data_transformers)
    val_set = FloorPlanDataset(image_dir=configuration.validation_data_config.validation_set_dir,gt_dir=configuration.validation_data_config.validation_ground_truth_dir,transform=data_transformers)

    #Loading train and val set
    train_set_loader = DataLoader(train_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)
    val_set_loader = DataLoader(val_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)

    #Build the model
    net = UNet(n_classes=3)

    if configuration.training_config.device.type == 'cuda':
        net.cuda()

    #loss_function = torch.nn.BCELoss()
    loss_function = torch.nn.CrossEntropyLoss()
    #loss_function = L.DiceLoss(mode="multiclass",classes=2)
    #loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=config.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    return net,train_set_loader,val_set_loader,loss_function,optimizer


def validation(nn_model,val_set_loader,loss_func):
    print("Validating.....")
    
    nn_model.eval()

    val_loss = 0.0
    mini_batches = 0

    for batch_id,(image,gt) in enumerate(val_set_loader):
        #image = image.squeeze(1)
        #image = image.reshape([image.shape[0],image.shape[-1],image.shape[2],image.shape[1]])
        
        #gt = gt.squeeze(1)
        #gt = gt.reshape([gt.shape[0],gt.shape[3],gt.shape[2],gt.shape[1]])
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
    wandb.watch(nn_model,log='all',log_freq=10)

    mini_batches = 0
    train_loss = 0.0
    print("Training....")
    for epoch in range(config.epochs):
        #scheduler.step()
        #print("\nLearning rate at this epoch is: %0.9f"%scheduler.get_lr()[0])
        for batch_id,(image,gt) in enumerate(train_set_loader):
            
            image = image.squeeze(1)
            image = image.long()
            gt = gt.squeeze(1)
            gt = gt.long()
            
            nn_model.train()
            if(configuration.training_config.device.type == 'cuda'):
                image,gt = image.to(device=configuration.training_config.device.type,dtype=torch.float),gt.to(device=configuration.training_config.device.type,dtype=torch.long)
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
                val_loss = validation(nn_model,val_set_loader,loss_func)
                training_log(val_loss,mini_batches)
                training_log(train_loss/mini_batches,mini_batches,False)

                PATH = "model.pt"
                torch.save({'epoch':epoch,'model_state_dict':nn_model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':train_loss},PATH)

                
                train_loss = 0.0

            print('Epoch-{0} lr:{1:f}'.format(epoch,optimizer.param_groups[0]['lr']))

def training_log(loss,mini_batch,train=True):
    if train == True:
        wandb.log({'batch':mini_batch,'loss':loss})
    else:
        wandb.log({'batch':mini_batch,'loss':loss})