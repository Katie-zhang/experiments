import PIL
import torch.utils.data as data
import medmnist
from medmnist import INFO
import torch
from torchvision import transforms as T
import numpy as np
from monai.transforms import Compose, Resize, ToTensor


def create_dataloaders(data_flag_list,batch_size, download=True):
    train_datasets_list = []
    val_datasets_list = []
    test_datasets_list = []
    train_dataloader_list = []
    val_dataloader_list = []
    test_dataloader_list = []
    
    for data_flag in data_flag_list:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        data_transform = get_transform(data_flag)
        
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
        val_dataset = DataClass(split='val', transform=data_transform,download=download, as_rgb=True)
        test_dataset = DataClass(split='test', transform=data_transform,download=download, as_rgb=True)
        
        train_datasets_list.append(train_dataset)
        val_datasets_list.append(val_dataset)
        test_datasets_list.append(test_dataset)
        
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)
        test_dataloader_list.append(test_dataloader)
    
    return (train_datasets_list, val_datasets_list, test_datasets_list,
            train_dataloader_list, val_dataloader_list, test_dataloader_list)

def get_num_classes_list(data_flag_list):
    num_classes_list = []
    for data_flag in data_flag_list:
        info = INFO[data_flag]
        num_classes = len(info['label'])
        num_classes_list.append(num_classes)
    return num_classes_list



def get_transform(data_flag):
    # check if 3D image
    if "3d" in data_flag:
        # apply 3D image transform
        return Compose([
            Resize(spatial_size=(224, 224, 112)),  # 假设这是你想要的 3D 尺寸
            ToTensor(dtype=torch.float32)
        ])
    else:
        # apply 2D image transform
        return T.Compose([
            T.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5])
        ])