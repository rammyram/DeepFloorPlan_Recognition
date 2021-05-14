import torch
import torchvision
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import configuration
from wandb_log import log
from dataloader import FloorPlan
from unet import UNet

def wandb_initializer(args):
    with wandb.init(project="Deepfloorplan_Recognition",config=args):
        config = wandb.config

        model,train_loader,val_loader,loss_func,optimizer = nn_model(config)
        train(model,train_loader,val_loader,loss_func,optimizer,config)
    return model

def nn_model(config):
    data_transformers = transforms.Compose([transforms.ToTensor()])

    train_set = 
    val_set = 

    #Loading train and val set
    train_set_loader = DataLoader(train_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)
    val_set_loader = DataLoader(val_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)

    #Build the model
    net = UNet(in_channels=1,out_channels=2)

    if configuration.training_config.device.type == 'cuda':
        net.cuda()

    loss_function = 

    optimizer = torch.optim.Adam(net.parmeters(),lr=config.lr)

    return net,train_set_loader,val_set_loader,loss_function,optimizer


def validation(nn_model,val_set_loader,loss_function):
    print("Validating....")
    nn_model.eval()

    val_loss = 0.0

    for batch_id,.. in enumerate(val_set_loader):
        if(configuration.training_config.device.type == 'cuda'):


        

        output = nn_model(...)
        loss = loss_function(...)

        mini_batches += 1
        val_loss += float(loss)

        val_loss = val_loss/mini_batches

        return val_loss


def train(nn_model,train_set_loader,val_set_loader,loss_func,optimizer,config):
    wandb.watch(nn_model,loss_function,log='all',log_freq=50)

    mini_batches = 0
    train_loss = 0.0

    for epoch in range(config.epochs):
        for batch_id,... in enumerate(train_set_loader):
            nn_model.train()
            if(configuration.training_config.device.type == 'cuda'):


            output = nn_model(...)
            loss = loss_func(...)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mini_batches += 1
            train_loss += float(loss)


            #Plotting in wandb
            if(mini_batches % configuration.training_config.plot_frequency == 0):
                val_loss = validation(nn_model,val_set_loader,loss_func)
                log(val_loss,mini_batches)
                log(train_loss,mini_batches,False)

                PATH = "model.pt"
                torch.save({'epoch':epoch,'model_state_dict':nn_model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':train_loss},PATH)

                
                train_loss = 0.0

            print('Epoch-{0} lr:{1:f}'.format(epoch,optimizer.param_groups[0]['lr']))