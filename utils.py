import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
from metrics import f1_score

# Get cpu or gpu device for training.
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from torch.functional import split



def train_(dataloader, model, loss_fn, optimizer,logger,dataset_test=None,ftt=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X,None) if ftt else model(X) 
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            yhat = torch.sigmoid(pred).round()
            f1_train = f1_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
            if(dataset_test is not None):
                f1_test = test(dataset_test,model,loss_fn,ftt)
                logger.log({"loss":loss,"F1_train":f1_train,"F1_val":f1_test})
            else:
                logger.log({"loss":loss,"F1_train":f1_train})
        del X,y,pred,loss

def train(dataloader, model, loss_fn, optimizer,dataset_test=None,ftt=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X,None) if ftt else model(X) 
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

def test(dataloader, model, loss_fn,ftt=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #print("num batches: "+str(num_batches))
    #print("size: "+str(size))
    model.eval()
    f1 = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X,None) if ftt else  model(X ) 

            pred = torch.sigmoid(pred)

            yhat = pred.round()
            # calculate accuracy
            f1 += f1_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
    f1 /= num_batches
    return f1
    