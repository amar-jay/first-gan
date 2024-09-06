import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Define transformations for high and low resolution images
transform_high_res = transforms.Compose([
    transforms.Resize(64),  # Resize to high resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_low_res = transforms.Compose([
    transforms.Resize(32),  # Resize to low resolution (original CIFAR10 size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class (as shown in the initial example)
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train, transform_high_res, transform_low_res):
        self.root = root
        self.train = train
        self.transform_high_res = transform_high_res
        self.transform_low_res = transform_low_res
        self.cifar10_dataset = datasets.CIFAR10(root=self.root, train=self.train, download=True)

    def __getitem__(self, index):
        image, _ = self.cifar10_dataset[index]
        
        # Apply high resolution transformation
        high_res_image = self.transform_high_res(image)
        
        # Apply low resolution transformation
        low_res_image = self.transform_low_res(image)
        
        return low_res_image,high_res_image 

    def __len__(self):
        return len(self.cifar10_dataset)

# Create custom dataset
train_dataset = CIFAR10Dataset(root='./data', train=True, transform_high_res=transform_high_res, transform_low_res=transform_low_res)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


# OLD CODE
def get_dataloader_(batch_size=64, num_workers=8):

    #TODO: write a better way to perform dataset random split
    train_dataset = CIFAR10Dataset(root='./data', train=True, transform_high_res=transform_high_res, transform_low_res=transform_low_res)
    val_dataset = CIFAR10Dataset(root='./data', train=True, transform_high_res=transform_high_res, transform_low_res=transform_low_res)
    test_dataset = CIFAR10Dataset(root='./data', train=True, transform_high_res=transform_high_res, transform_low_res=transform_low_res)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    return train_loader, val_loader, test_loader

def get_dataloader(batch_size=64, num_workers=8):
    dataset = CIFAR10Dataset(root='./data', train=True, transform_high_res=transform_high_res, transform_low_res=transform_low_res)

# Define the sizes for train, val, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

# Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(num_workers=1, batch_size=4)
    print(f"{len(train_loader)=},\t\t{len(val_loader)=},\t\t{len(test_loader)=}")

    for batch_idx, batch in enumerate(train_loader):
        high_res_images, low_res_images = batch
        print(f"{high_res_images.shape=},\t{low_res_images.shape=}")
        break

