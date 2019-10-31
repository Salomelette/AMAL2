
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import torch.nn as nn
#import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datamaestro import prepare_dataset
import torch.nn.functional as F

#Dataset et Dataloader

class MonDataset(Dataset) :
    def __init__(self,data,label) :
        self.data=torch.tensor(data,dtype=torch.float)/255
        self.label=torch.tensor(label,dtype=torch.float)
    def __getitem__(self , index ) :  
        return self.data[index], self.label[index]
        #mettre image sous forme de vecteur
    #retourne un couple  exemple , l a b e l
    def __len__(self) :  
        return list(self.label.size())[0]


ds = prepare_dataset ( "com.lecun.mnist")
train_images ,  train_labels = ds.files["train/images"].data() ,ds.files["train/labels"].data()
test_images ,  test_labels = ds.files["test/images"].data() , ds.files["test/labels"].data()

BATCH_SIZE = 5
data_train = DataLoader(MonDataset(train_images, train_labels), shuffle=True ,  batch_size=BATCH_SIZE)

data_test = DataLoader(MonDataset(test_images, test_labels), shuffle=True ,  batch_size=BATCH_SIZE)



#Autoencodeur
"""
class Encodeur(nn.Module):
    def __init__(self,dim_in,dim_out) :
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(dim_in,dim_out)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        y = self.linear(x)
        return self.relu(y)

class Decodeur(nn.Module):
    def __init__(self,dim_in,dim_out) :
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(dim_in,dim_out)
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        y = self.linear(x).squeeze()
        return self.relu(y)

"""

class Autoencodeur(nn.Module):
    def __init__(self,dim_in,dim_out) :
        super(Model,self).__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.w= torch.nn.Parameter(torch.randn(self.dim_in,self.dim_out,dtype=torch.float64))
        self.b=torch.nn.Parameter(torch.randn(1,dtype=torch.float64))

    def encodeur(self,x):
        model = torch.nn.Sequential(torch.nn.functionnal.linear(x, self.w,self.b),torch.nn.functionnal.relu())
        model.train()
        return model(x.float())

    def decodeur(self,y) :
        
        self.sig = torch.nn.Sigmoid()
        model = torch.nn.Sequential(torch.nn.functionnal.linear(y, self.w.T,self.b),torch.nn.functionnal.sigmoid())
        model.train()
        return model(y.float())

encode = Encodeur(28,5)
for x, y in data_train :
    pass
#A tester
#Faire parameter et sequencial, function