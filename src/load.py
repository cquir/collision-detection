import numpy
import torch

class Dataset(torch.utils.data.Dataset):
	def __init__(self,dataset,label,Nfolds,bdir):
		self.X = []; self.Y = []; self.Nfolds = Nfolds
		for idx in range(Nfolds):
			fold = '' if Nfolds == 0 else f'-fold-{idx}'
			self.X.append(numpy.loadtxt(f'{bdir}/data/datasets/{label}/X{dataset}-{label}{fold}.dat').astype(numpy.float32))
			self.Y.append(numpy.loadtxt(f'{bdir}/data/datasets/{label}/Y{dataset}-{label}{fold}.dat').astype(numpy.float32))
	def __len__(self):
		return self.Nfolds*len(self.X[0])
	def __getitem__(self,idx):
		fold = idx // len(self.X[0]); idx = idx % len(self.X[0])
		return (self.X[fold][idx],self.Y[fold][idx])

def load_data(args):
	bdir = '..'
	split = args['split']; Ntot = args['dataset_examples']
	Nfolds = max(1,int(Ntot/5e5))
	train_data = Dataset('train',f'rel-in-split-{split}-Ntot-{Ntot:.1e}',Nfolds,bdir)
	val_data = Dataset('validation',f'rel-in-split-{split}-Ntot-{Ntot:.1e}',Nfolds,bdir)
	train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['test_batch_size'],shuffle=True)
	return train_loader,val_loader
