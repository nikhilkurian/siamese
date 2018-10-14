import torch.nn as nn
from torchvision import models
original_model = models.resnet18(pretrained=True)
num_ftrs=8192
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.xfcn = nn.Sequential(
                    
                    *list(original_model.children())[:-1]
                )
        self.fcn1=nn.Sequential(nn.Linear(num_ftrs,4096 ),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096,1024))

    def forward_once(self, x):
        output = self.xfcn(x)
        output = output.view(output.size()[0], -1)
        output = self.fcn1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2