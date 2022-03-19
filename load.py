import numpy
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,label):
        X = numpy.loadtxt('data/X%s.dat'%label).astype(numpy.float32)
        Y = numpy.loadtxt('data/Y%s.dat'%label).astype(numpy.float32)
        self.data = (X,Y)
    def __len__(self):
        return len(self.data[1])
    def __getitem__(self,idx):
        return (self.data[0][idx],self.data[1][idx])

def load_data(args):
    train_data = Dataset('train')
    val_data = Dataset('validation')
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['test_batch_size'],shuffle=True)
    return train_loader,val_loader
