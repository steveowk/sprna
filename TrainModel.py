import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import math
import os
from os.path import exists

class Model():
    def __init__(self,
    model,
    epochs,
    optimizer,
    criterion,
    model_filename,
    batch_size,
    lr,
    after,
    global_step
    ):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.epochs = epochs
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.criterion = self.get_loss(criterion)
        self.regression = True
        self.bestloss = math.inf
        self.model_filename = model_filename    
        self.load_model()  
        self.after = after  
        self.global_step = global_step      
    
    def get_loss(self,criterion):
        if criterion == "mse":return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":return Adam(self.model.parameters(), lr = lr)
        return SGD(self.model.parameters(), lr = lr, momentum = 0.9)
    
    def load_model(self):
        if exists(self.model_filename):
            checkpoint = torch.load(self.model_filename)
            self.model.load_state_dict(checkpoint['model'])
            print(f"model loaded successfylly")

    def trainer(self, train_loader, valid_loader):    
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        for epx in range(self.epochs):
            self.train(epx)    
            self.test(epx)
    
    def train(self, epoch):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.float()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()    
        
    def test(self,epoch):
        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
            inputs = inputs.float()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.detach().item()
            #if self.bestloss>loss:
            self.bestloss = loss
            state = {
                'model': self.model.state_dict(),
                'loss': self.bestloss,
            }
            torch.save(state, self.model_filename)
        if epoch%self.after==0:print(f"End of Epoch {epoch} loss {loss} best {self.bestloss}")
    
    def predict(self,x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x).item()