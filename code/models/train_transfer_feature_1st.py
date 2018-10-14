#reshaping will be batchsizexrestof dimen
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

class ResNet18Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet18Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        print(self.features)
        
    def forward(self, x):
        x = self.features(x)
        x=x.view(10,-1)
        out=nn.Linear(x.size()[1],2)
        return out
        
# Data augmentation and normalization for training
# Just normalization for validation
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
b_num=2706.0
m_num=5850.0
m_by_b=m_num/b_num
data_dir = '/home/vindhya/Documents/Siamese_Project/Datas/new_expt_1_train_val_breakhis_psep/'
#data_dir='/home/vindhya/Documents/Siamese_Project/Datas/expt_1_train_Bach_test_BreakHis'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), #Make a dataset object, with __getitem__()(or []) returns tuples and len() or __len__()
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, #make the dataset object iterable(not iterant) with the given batch size , it has len methods but no __getitem__() to access elements make it iterant by __iter__() or iter() and call __next__() or next()
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # This will extract the folder names from the file structure and name them as class names

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights=[m_by_b,1]
class_weights = torch.FloatTensor(weights).cuda()
# print(dataset_sizes)

def train_model(model,model_sz, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.,iter(dataloaders['train']) gives a list of 2 elements , if we pass to two element then 1st variable is tensor of images with batch size and second is tensor of labels
            for inputs, labels in dataloaders[phase]: #here inputs are batch sized tensors and labels are tensors of size 1xbatchsize
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model_sz(inputs)
                    outputs = model(inputs) #output is batchsizex2 tensor hence maximum has to be sampled acros the column
                    # print(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # print("The loss is ")
                    # print(loss)
                    # print("The loss item is")
                    # print(loss.item())

 
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() #check how new loss incorporates backward() etc
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# model_ft = models.resnet18(pretrained=True)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
print("The model number of features are {}".format(num_ftrs))

#res50_model = models.resnet50(pretrained=True)
# res50_conv2 = ResNet50Bottom(res50_model)
model_ft_sz = ResNet18Bottom(model_ft)
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)



model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
model_ft_sz=model_ft_sz.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights) #callable object, use callable to see, callable because of forward function is overwritten
# print(criterion)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft,model_ft_sz, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

