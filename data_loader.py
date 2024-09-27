import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config
import random
import numpy as np

def set_seed(seed=config.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ])
    return train_transform, val_transform

def get_data_loaders():
    set_seed()
    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(root=os.path.join(config.data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config.data_dir, 'val'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return train_loader, val_loader
