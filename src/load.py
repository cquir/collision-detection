import numpy
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,label,bdir):
        X = numpy.loadtxt(f'{bdir}/data/datasets/X{label}.dat').astype(numpy.float32)
        Y = numpy.loadtxt(f'{bdir}/data/datasets/Y{label}.dat').astype(numpy.float32)
        self.data = (X,Y)
    def __len__(self):
        return len(self.data[1])
    def __getitem__(self,idx):
        return (self.data[0][idx],self.data[1][idx])

def load_data(args):
    bdir = '..'
    train_data = Dataset('train',bdir)
    val_data = Dataset('validation',bdir)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['test_batch_size'],shuffle=True)
    return train_loader,val_loader
