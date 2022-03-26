import subprocess
import numpy
import torch
import train
import load
import sys
import os

args = dict(
    i = 2*3+2*4,
    dropout = False,
    early_stopping = False,
    test_batch_size = 1,
    log_interval = 1,
    epochs = 200,
    patience = 30,
    seed = 0,
    save = False,
    validation = False
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

# update hyperparameters
args['hidden_layers'] = 1
args['h'] = 500
args['lr'] = 3e-4
args['batch_size'] = 64

# identifier
args['ID'] =  'hidden_layers_{}_h_{}_lr_{:.1e}_batch_size_{}'.format(
    args['hidden_layers'],args['h'],args['lr'],args['batch_size']) 

# print hyperparameters
print('hidden layers: {}, h: {}, learning rate: {:.4e}, batch size: {}\n'.format(
    args['hidden_layers'],args['h'],args['lr'],args['batch_size'])) 
sys.stdout.flush()

# define dataloaders
bdir = '/home/cquir/Documents/collision-detection'
train_data = load.Dataset('tiny_train',bdir)
val_data = load.Dataset('validation',bdir)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=args['test_batch_size'],shuffle=True)

# training loop
model, train_losses, val_losses, train_accuracies, val_accuracies = train.evaluate_model(args,train_loader,val_loader)
