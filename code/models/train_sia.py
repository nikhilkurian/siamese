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
                    # lbl=0
                    break
        else:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]!=img1_tuple[1]:
                    # lbl=1
                    break

            #img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

      
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        #return img0, img1 , torch.from_numpy(np.array(lbl,dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs*10)

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
                        batch_size=10) for x in ['train', 'val']}
train_dataloader=dataloaders['train']


dataset_sizes = {x: len(siamese_dataset[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # This will extract the folder names from the file structure and name them as class names

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# weights=[m_by_b,1]
# class_weights = torch.FloatTensor(weights).cuda()
# print(dataset_sizes)

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def train_model(model, criterion, optimizer,scheduler, num_epochs=24):
    counter = []
    loss_history = [] 
    iteration_number= 0

    model.train()
    for epoch in range(0,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            img0, img1 , label = data
            img0, img1 , label = Variable(img0.to(device)), Variable(img1.to(device)) , Variable(label.to(device))
            optimizer.zero_grad()
            output1,output2 = model(img0,img1)
            loss_contrastive = criterion(output1,output2,label).cuda()
            running_loss +=loss_contrastive.item()* img0.size(0)
            loss_contrastive.backward()
            optimizer.step()
            if i %100 == 0 :
                print(i)
            #     print("iter number {}\n Current loss {}\n".format(iteration_number,loss_contrastive.item()))
            #     iteration_number +=100   
            #     counter.append(iteration_number)
            #     loss_history.append(running_loss)
        loss_norm=running_loss/dataset_sizes['train']
        print("Epoch number {}\n Current loss {}\n".format(epoch,running_loss))        
        counter.append(epoch)
        loss_history.append(running_loss)
    show_plot(counter,loss_history)
        # Each epoch has a training and validation phase
        # for phase in ['train']:
        #     if phase == 'train':
        #         scheduler.step()
        #         model.train()  # Set model to training mode
        #     # else:
        #     #     model.eval()   # Set model to evaluate mode

        #     running_loss = 0.0
        #     # running_corrects = 0
        #     for i, data in enumerate(dataloaders[phase],0):
        #         img0, img1 , label = data
        #         img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
        #         optimizer.zero_grad()
                
        #         loss = criterion(output1,output2,label)
        #         loss.backward()
        #         optimizer.step()
        #         with torch.set_grad_enabled(phase == 'train'):
        #             output1,output2 = model(img0,img1) #output is batchsizex2 tensor hence maximum has to be sampled acros the column
        #             # print(outputs)
        #             # _, preds = torch.max(outputs, 1)
        #             loss = criterion(output1,output2,label)
                    # loss = criterion(outputs, labels)
            # Iterate over data.,iter(dataloaders['train']) gives a list of 2 elements , if we pass to two element then 1st variable is tensor of images with batch size and second is tensor of labels
            # for inputs, labels in dataloaders[phase]: #here inputs are batch sized tensors and labels are tensors of size 1xbatchsize
            #     inputs = inputs.to(device)
            #     labels = labels.to(device)

            #     # zero the parameter gradients
            #     optimizer.zero_grad()

                # forward
                # track history if only in train

                    # print("The loss is ")
                    # print(loss)
                    # print("The loss item is")
                    # print(loss.item())

 
                    # backward + optimize only if in training phase
    #                 if phase == 'train':
    #                     loss.backward() #check how new loss incorporates backward() etc
    #                     optimizer.step()

    #             # statistics
    #             running_loss += loss.item() * inputs.size(0)
    #             # running_corrects += torch.sum(preds == labels.data)

    #         epoch_loss = running_loss / dataset_sizes[phase]
    #         # epoch_acc = running_corrects.double() / dataset_sizes[phase]

    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #             phase, epoch_loss, epoch_acc))

    #         # deep copy the model
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())

    #     print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model


def main():

# model_ft = models.resnet18(pretrained=True)
# model_ft = models.resnet18(pretrained=True)
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    model_ft =SiameseNetwork()
# num_ftrs = model_ft.fc.in_features

# model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights) #callable object, use callable to see, callable because of forward function is overwritten
# print(criterion)
# Observe that all parameters are being optimized
    criterion = ContrastiveLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(),lr = 0.00005) 
# Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,
                           num_epochs=24)
if __name__ == '__main__':
    main()
