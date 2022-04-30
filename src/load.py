import numpy
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset,label,bdir):
        X = numpy.loadtxt(f'{bdir}/data/datasets/{label}/X{dataset}-{label}.dat').astype(numpy.float32)
        Y = numpy.loadtxt(f'{bdir}/data/datasets/{label}/Y{dataset}-{label}.dat').astype(numpy.float32)
        self.data = (X,Y)
    def __len__(self):
        return len(self.data[1])
    def __getitem__(self,idx):
        return (self.data[0][idx],self.data[1][idx])

def load_data(args):
    bdir = '..'
    split = args['split']; Ntot = args['dataset_examples']
    train_data = Dataset('train',f'rel-in-split-{split}-Ntot-{Ntot:.1e}',bdir)
    val_data = Dataset('validation',f'rel-in-split-{split}-Ntot-{Ntot:.1e}',bdir)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['test_batch_size'],shuffle=True)
    return train_loader,val_loader
