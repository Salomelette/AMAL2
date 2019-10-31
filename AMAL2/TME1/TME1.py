# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
import numpy as np
from datamaestro import prepare_dataset 
import random

class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class Linear(Function):

    @staticmethod
    def forward(ctx,x,w,b):
        #input = [x,w,b]
        ctx.save_for_backward(x,w,b)
        return torch.dot(x,w) + b

    @staticmethod
    def backward(ctx,grad_output):
        x,w,b = ctx.saved_tensors
        Lx = grad_output*w
        Lw = grad_output*x
        return Lx, Lw, grad_output


## Exemple d'implementation de fonction a 2 entrÃ©es
class MSE(Function):

    @staticmethod
    def forward(ctx,y,y_c):
        ctx.save_for_backward(y,y_c)
        d=(y-y_c).pow(2).mean()
        return d

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrÃ©es
        y,y_c = ctx.saved_tensors
        d=2*(y-y_c).mean()
        return d*grad_output

## Pour utiliser la fonction 
L = Linear()
ctx=Context()

x = torch.randn(5,requires_grad=True,dtype=torch.float64)
w = torch.randn(5,requires_grad=True,dtype=torch.float64)
b = torch.randn(1,requires_grad=True,dtype=torch.float64)
f=L.forward(ctx,x,w,b)
L_gradcheck=L.apply #check pour utiliser requires_grad

torch.autograd.gradcheck(L_gradcheck,(x,w,b)) #verifie la coherence du gradient

#faire split avec sklearn
def split(X,Y,pct):
    index=np.arange(len(X))
    random.shuffle(index)
    trainx=X[index[:int(pct*len(index))]]
    trainy=Y[index[:int(pct*len(index))]]
    testx=X[index[int(pct*len(index)):]]
    testy=Y[index[int(pct*len(index)):]]
    return torch.from_numpy(trainx),torch.from_numpy(trainy),torch.from_numpy(testx),torch.from_numpy(testy)
    

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data() 

trainx,trainy,testx,testy=split(data.T[:-1].T,data.T[-1].T,pct=0.8)

#voir solutions sur le tuto de pytorch comparer resultats

MSE=MSE()
ctxMse=Context()

N=10000
learning_rate= 10e-3
batch=20 #taille des batch


w = torch.randn((1,trainx.shape[1]),requires_grad=True,dtype=torch.float64) #dim de X
b = torch.randn((1,1),requires_grad=True,dtype=torch.float64)

#Descente de gradient pour apparendre w,b
for i in range(N):

    index_batch =random.randint(0,len(trainx)-1-batch)
    x=trainx[index_batch:index_batch+batch]
    y=trainy[index_batch:index_batch+batch]
    
    l=torch.mm(x,w.T)+b
    mse=(l - y).pow(2)
    mse.mean().backward()

    with torch.no_grad(): #descente de gradient
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad


print(w,b)