# tests/test_data_loader.py
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection

def collate_fn(batch):
    images, targets = zip(*batch)  
    images = torch.stack(images)  # Empilha as imagens em um tensor
    return images, targets 

def test_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = CocoDetection(
        root="./dataset/train/",           
        annFile="./dataset/train/_annotations.coco.json", 
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    data_iter = iter(loader)
    images, targets = next(data_iter)
    
    assert images.shape == (8, 3, 224, 224) 
    assert isinstance(targets, tuple)  # Targets devem ser uma tupla de listas de anotações
    assert len(targets) == 8  