from  datamaestro  import  prepare_dataset
from torch.utils.data import Dataset, DataLoader

ds = prepare_dataset("org.universaldependencies.french.gsd")
train, dev, test = (ds.files[n].data() for n in ("train", "dev", "test"))

class DataSetEtiquettage(DataSet):
    def __init__(self,data,labels):
        super(DatasetEtiquettage,self).__init__()
        self.data = data 
        self.labels = labels 

    def get_items(self,index):
        pass
