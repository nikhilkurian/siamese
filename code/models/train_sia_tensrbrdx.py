from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset
from fromresnet import SiameseNetwork
from contrastiveloss import ContrastiveLoss
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter


class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        # self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    lbl=0
                    break
        else:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]!=img1_tuple[1]:
                    lbl=1
                    break

            #img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

      
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        #return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        return img0, img1 , torch.from_numpy(np.array([lbl],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
}

data_dir = '/home/vindhya/Documents/Siamese_Project/Datas/new_expt_1_train_val_breakhis_psep/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), #Make a dataset object, with __getitem__()(or []) returns tuples and len() or __len__()
                                          data_transforms[x])
                  for x in ['train', 'val']}
siamese_dataset ={x:SiameseNetworkDataset(imageFolderDataset=image_datasets[x],
                                        transform=data_transforms['train']
                                        ) for x in ['train', 'val']}
dataloaders = {x:torch.utils.data.DataLoader(siamese_dataset[x],
                        shuffle=True,
                        num_workers=4,
                        batch_size=40) for x in ['train', 'val']}
train_dataloader=dataloaders['train']


dataset_sizes = {x: len(siamese_dataset[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # This will extract the folder names from the file structure and name them as class names

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()






def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def train_model(model, criterion, optimizer, num_epochs=24):
    counter = []
    loss_history = [] 
    #iteration_number= 0

    model.train()
    




    for i, data in enumerate(train_dataloader):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0.to(device)), Variable(img1.to(device)) , Variable(label.to(device))
        optimizer.zero_grad()
        output1,output2 = model(img0,img1)
        loss_contrastive = criterion(output1,output2,label).cuda()
        # running_loss +=loss_contrastive.item()* img0.size(0)
        loss_contrastive.backward()
        optimizer.step()
        # if i %100 == 0 :
        #     print(i)
            #     print("iter number {}\n Current loss {}\n".format(iteration_number,loss_contrastive.item()))
            #     iteration_number +=100   
            #     counter.append(iteration_number)
            #     loss_history.append(running_loss)
        # loss_norm=running_loss/dataset_sizes['train']
        print("Iteration number {}\n Current loss {}\n".format(i,loss_contrastive.item()))        
        counter.append(i)
        loss_history.append(loss_contrastive.item())
        writer.add_scalar('Loss',loss_contrastive.item(),i)
    show_plot(counter,loss_history)





def main():


    model_ft =SiameseNetwork()


    model_ft = model_ft.to(device)

    criterion = ContrastiveLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00005, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(),lr = 0.00005) 
# Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           num_epochs=24)
if __name__ == '__main__':
    main()
