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
from dataloader import FloorPlanDataset
from unet import UNet

def wandb_initializer(args):
    with wandb.init(project="Deepfloorplan_Recognition",config=args):
        config = wandb.config

        model,train_loader,val_loader,loss_func,optimizer = nn_model(config)
        train(model,train_loader,val_loader,loss_func,optimizer,config)
    return model

def nn_model(config):
    data_transformers = transforms.Compose([transforms.ToTensor()])

    train_set = FloorPlanDataset(image_dir=configuration.train_data_config.training_set_dir,door_dir=configuration.train_data_config. doors_training_ground_truth_dir,window_dir=configuration.train_data_config.windows_training_ground_truth_dir,transform=data_transformers)
    val_set = FloorPlanDataset(image_dir=configuration.validation_data_config.validation_set_dir,door_dir=configuration.validation_data_config. doors_validation_ground_truth_dir,window_dir=configuration.validation_data_config.windows_validation_ground_truth_dir,transform=data_transformers)

    #Loading train and val set
    train_set_loader = DataLoader(train_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)
    val_set_loader = DataLoader(val_set,batch_size = configuration.training_config.batch_size,shuffle=False,num_workers=configuration.training_config.number_workers)

    #Build the model
    net = UNet(in_channels=1,out_channels=2)

    if configuration.training_config.device.type == 'cuda':
        net.cuda()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1,reduction='mean')

    optimizer = torch.optim.Adam(net.parameters(),lr=config.lr)

    return net,train_set_loader,val_set_loader,loss_function,optimizer


def validation(nn_model,val_set_loader,loss_function):
    print("Validating....")
    nn_model.eval()

    val_loss = 0.0

    for batch_id,(image,door,window) in enumerate(val_set_loader):
        if(configuration.training_config.device.type == 'cuda'):
            image,door,window = image.cuda(), door.cuda(), window.cuda()
        else:
            image,door,window = image, door, window


        output = nn_model(image.float())
        loss_door = loss_func(output[0].float(), door.float())
        loss_window = loss_func(output[1].float(),window.float())

        mini_batches += 1
        val_loss += float(loss)

        val_loss = val_loss/mini_batches

        return val_loss


def train(nn_model,train_set_loader,val_set_loader,loss_func,optimizer,config):
    wandb.watch(nn_model,loss_func,log='all',log_freq=50)

    mini_batches = 0
    train_loss = 0.0
    print("Training....")
    for epoch in range(config.epochs):
        for batch_id,(image,door,window) in enumerate(train_set_loader):
            nn_model.train()
            if(configuration.training_config.device.type == 'cuda'):
                image,door,window = image.cuda(), door.cuda(), window.cuda()
            else:
                image,door,window = image, door, window

            output = nn_model(image.float())
            loss_door = loss_func(output[0].float(), door.float())
            loss_window = loss_func(output[1].float(),window.float())
            
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

def training_log(loss,mini_batch,train=True):
    if train == True:
        wandb.log({'batch':mini_batch,'loss':loss})
    else:
        wandb.log({'batch':mini_batch,'loss':loss})