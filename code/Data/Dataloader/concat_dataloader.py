from torchvision import datasets, models, transforms
import torch

traindir_A="/home/vindhya/Documents/Siamese_Project/Dataset/Bach/cancerous_patches200"
traindir_B="/home/vindhya/Documents/Siamese_Project/Dataset/Bach/Non_cancerous_patches200"

# train_loader_A = torch.utils.data.DataLoader(
#              datasets.ImageFolder(traindir_A),
#              batch_size=10, shuffle=True,
#              num_workers=4, pin_memory=True)

train_loader_A = torch.utils.data.DataLoader(
             datasets.ImageFolder(traindir_A),
             batch_size=10, shuffle=True,
             num_workers=4)

train_loader_B = torch.utils.data.DataLoader(
             datasets.ImageFolder(traindir_B),
             batch_size=10, shuffle=True,
             num_workers=4)



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(traindir_A),
                 datasets.ImageFolder(traindir_B)
             ),
             batch_size=args.batch_size, shuffle=True,
             num_workers=args.workers, pin_memory=True)



