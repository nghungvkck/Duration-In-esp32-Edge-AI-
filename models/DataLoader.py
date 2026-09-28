import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DataLoader:
    # kiểm tra GPU có sẵn 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device = {device}")
    
    train_dataset  = TensorDataset(torch.load('data/train_data.pt'), torch.load('data/train_labels.pt'))
    val_dataset    = TensorDataset(torch.load('data/val_data.pt'), torch.load('data/val_labels.pt'))
    test_dataset   = TensorDataset(torch.load('data/test_data.pt'), torch.load('data/test_labels.pt'))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    