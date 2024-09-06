from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Custom dataset to create low-resolution images from high-resolution images
class SuperResolutionVAEDataset(Dataset):
    def __init__(self, hr_images, scale_factor=2):
        self.hr_images = hr_images
        self.scale_factor = scale_factor
        self.lr_transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.Resize((64 // scale_factor,  64// scale_factor), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] range
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] range
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image, _ = self.hr_images[idx]
        hr_image = Image.open(hr_image)
        lr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

def get_dataloader(batch_size=64, num_workers=8):

    #TODO: write a better way to perform dataset random split
    dset = datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = SuperResolutionVAEDataset(dset)
    test_dataset = SuperResolutionVAEDataset(datasets.CIFAR10(root='./data', train=False, download=True))
    val_dataset = SuperResolutionVAEDataset(datasets.CIFAR10(root='./data', train=False, download=True))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(num_workers=1, batch_size=4)
    print(len(train_loader), len(val_loader), len(test_loader))
    for data in train_loader:
        inputs, labels = data
        print(inputs.shape)  # Check the shape to confirm the data is loaded correctly
        break
