import torch
from torch.autograd import Function
import numpy as np
import pandas as pd
import torch.nn as nn
import pdb


class RNN(nn.Module):
    def __init__(self, length, batch, dim, latent, dimout):
        super().__init__()
        self.latent = latent
        self.length = length
        self.batch = batch
        self.dim = dim
        self.dimout = dimout
        self.linear = torch.nn.Linear(self.dim,self.latent).double() #a revoir un peu 
        self.hidden = torch.nn.Linear(self.latent,self.latent).double() 
        self.hidden_out = torch.nn.Linear(self.latent,self.dimout).double()
        self.tanh = torch.nn.Tanh()

    def one_step(self,x,h): #x de dim b*d, h de dim b*l
        print(x.size())
        pdb.set_trace() 
        x_t = self.linear(x)
        h_t = self.hidden(h)
        y_t = self.tanh(x_t + h_t)
        return y_t

# one_step_final au cas ou 

    def forward(self,x,h): #x de dim l*b*d, h de dim b*l 
        res = torch.zeros((self.length,self.batch,self.latent)).double()
        for i,elem in enumerate(x):
            res[i] = h 
            new_h = self.one_step(elem,h)
            h = new_h.clone()  
        y = self.hidden_out(res[-1])
        return res,y


if __name__ == '__main__':

    seqSize = 50

    df = pd.read_csv('tempAMAL_train.csv')
    col_names = df.columns.values[1:]
    train_x = []
    train_y = []
    labels = []
    for k,city in enumerate(col_names):
        d = np.array(df[city])
        d = np.where(np.isfinite(d), np.nanmean(d,axis=0), d)
        for i in range(0,len(d)-seqSize,seqSize):
            train_x.append(d[i:i+seqSize])
            train_y.append(k)
        labels.append(city)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_x = torch.from_numpy(train_x).double()
    train_y = torch.from_numpy(train_y).double()

    #A FINIR, site super cool : https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/ à consulter sans moderation ma poule  

    batch_size = 100
    hidden_size = 50
    learning_rate = 0.001
    dimout = 1

    batch_train_x = []
    batch_train_y = []
    #shuffle a faire  
    for i in range(0,train_x.size()[0]-batch_size,batch_size):
        batch_train_x.append(train_x[i:i+batch_size]) #transpose ?
        batch_train_y.append(train_y[i:i+batch_size])

   
    model = RNN(seqSize,batch_size,1,hidden_size,dimout) #.to(device) ?
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

    nb_epochs = 1000

    for k in range(nb_epochs):
        tab_loss = []
        for batch_x,batch_y in zip(batch_train_x,batch_train_y):
            model.train()
            h = torch.zeros(batch_size,hidden_size,requires_grad=True).double()
            print(batch_x.size())
            pdb.set_trace()
            _,ypred = model(batch_x.double(),h)
            _,indy = torch.max(ypred,1)
            loss = criterion(ypred,indy) 
            loss.backward()
            tab_loss.append(loss)

        optim.step()
        optim.zero_grad()
        if k%10==0:
            print("epoch={} loss={}".format(k,np.mean(tab_loss)))
            





    