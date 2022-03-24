import subprocess
import numpy
import torch
import train
import load
import sys
import os

args = dict(
    subdir = 'low-level-features',
    test_batch_size = 1000,
    log_interval = 1000,
    epochs = 100,
    patience = 10,
    i = 2*3+2*4,
    dropout = False,
    seed = 0
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

N = 100
for i in range(N):
    # update + print hyperparameters
    args['hidden_layers'] = int(sys.argv[1])
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=4*[0.25]))[0][0]
    args['h'] = [50,100,200,500][idx]
    args['lr'] = 10**numpy.random.uniform(low=-5,high=-1)
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=5*[0.2]))[0][0]
    args['batch_size'] = [32,64,128,256,512][idx]
    print('[{}/{}] hidden layers: {}, h: {}, learning rate: {:.4e}, batch size: {}\n'.format(
        i,N,args['hidden_layers'],args['h'],args['lr'],args['batch_size'])) 
    sys.stdout.flush()

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    model, train_loss, val_loss, val_accuracy = train.evaluate_model(args,train_loader,val_loader)

    # save results
    subdir = args['subdir']
    s =  'hidden_layers_{}_h_{}_lr_{:.1e}_batch_size_{}'.format(
        args['hidden_layers'],args['h'],args['lr'],args['batch_size']) 
    if s not in os.listdir('data/'):
        subprocess.call(f'mkdir data/results/{subdir}/{s}',shell=True)
    header = 'args = {}'.format(repr(args))
    numpy.savetxt(f'data/results/{subdir}/{s}/train_loss_{s}.dat',train_loss,header=header)
    numpy.savetxt(f'data/results/{subdir}/{s}/val_loss_{s}.dat',val_loss,header=header)
    numpy.savetxt(f'data/results/{subdir}/{s}/val_accuracy_{s}.dat',val_accuracy,header=header)
    torch.save(model.state_dict(),f'data/results/{subdir}/{s}/model_{s}.pt')
