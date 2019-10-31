
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
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

device = 'cpu'
dtype = torch.float
x = torch.randn((1,10),requires_grad=True,dtype=torch.float,device=device)
w = torch.randn((1,10),requires_grad=True,dtype=torch.float,device=device)
#b = torch.randn(1,1,requires_grad=True,dtype=torch.float64)
y = torch.randint(2,size=(1,),dtype=torch.float,device='cpu')

learning_rate = 10e-3
#data, target = load_boston(return_X_y=True)
#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)



""" for i in range(200):
    y_pred = x.mm(w.T)
    mse = (y_pred-y).pow(2)
    mse.backward()
    print("y_pred :",y_pred," y_true :",y," loss :",mse)
    with torch.no_grad():
        w -= learning_rate*w.grad
        w.grad.zero_() """

""" optim = torch.optim.SGD(params=[w],lr=0.001)

for i in range(50):
    y_pred = x.mm(w.T)
    mse = (y_pred-y).pow(2)
    mse.backward()
    optim.step()
    optim.zero_grad() """

class Linear(Function):

    @staticmethod
    def forward(ctx,x,w,b):
        input = [x,w,b]
        ctx.save_for_backward(input)
        return torch.dot(x,w) + b

    @staticmethod
    def backward(ctx,grad_output):
        x,w = ctx.saved_tensors
        Lx = torch.dot(grad_output,w) #grad_output*w
        Lw = torch.dot(grad_output,x) #grad_output*x
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

class Linear2(torch.nn.Module):

    def __init__(self,dim_in,dim_out):#on peut ajouter dim de couche cachee
        super(Wiwi,self).__init__()
        self.linear = torch.nn.Linear(dim_in,dim_out)

    def forward(self,x):
        y = self.linear(x).squeeze()
        return y 
    #on fait passer y dans les couches



#utiliser tensor board



## Pour utiliser la fonction 
L = Linear1()
ctx=Context()

x = torch.randn(10,5,requires_grad=True,dtype=torch.float64)
w = torch.randn(1,5,requires_grad=True,dtype=torch.float64)
b = torch.randn(1,requires_grad=True,dtype=torch.float64)
f=output = L.forward(ctx,x,w,b)
L_gradcheck=L.apply #check pour utiliser requires_grad

torch.autograd.gradcheck(L_gradcheck,(x,w,b)) #verifie la coherence du gradient

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
loss2 = torch.nn.MSELoss()
ctxMse=Context()

N=10000
learning_rate = 10e-3

w = torch.randn((1,trainx.shape[1]),requires_grad=True,dtype=torch.float64) #dim de X
b = torch.randn((1,1),requires_grad=True,dtype=torch.float64)
model = Linear2(trainx.size()[1],trainx.size()[0])
optim = torch.optim.Adam(model.parameters(),lr=learning_rate) #pas de gradient
"""
Equivalent à :  mettre a la place de l'aute si compile pas 
w= torch.nn.Parameter(torch.randn(1,trainx.shape[1],dtype=torch.float64))
b=torch.nn.Parameter(torch.randn(1,dtype=torch.float64))
optim = torch.optim.Adam(params=[w,b],lr=lr)
"""
#GRADIENT STOCHASTIQUE
#Descente de gradient pour apparendre w,b
for i in range(N):

    index_batch =random.randint(0,len(trainx)-1-batch)
    x=trainx[index_batch]
    y=trainy[index_batch]
    
    l=torch.mm(x,w.T)+b
    mse=(l - y).pow(2) #faire loss.backward
    mse.mean().backward()

    if i %100 ==0:
        optim.step()
        optim.zero_grad()


print(w,b)
#graditn stohastique : 1 exemple alors que batch size paquet e talle bath doc batch fois plus rapide