import subprocess
import numpy
import torch
import run
import sys
import os

if not os.path.isdir("./data/results"):
    os.mkdir("./data/results")

args = dict(
    i = 2*3+2*4,
    dropout = False,
    early_stopping = False,
    test_batch_size = 1000,
    log_interval = 1000,
    epochs = 200,
    patience = 30,
    seed = 0,
    save = True,
    validation = False,
    dataset="data/datasets/smoketest"
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

rand_idx = lambda ln: numpy.nonzero(numpy.random.multinomial(n=1,pvals=ln*[1./ln]))[0][0]

N = 1000
for i in range(N):

    # update + print hyperparameters
    args['hidden_layers'] = numpy.arange(1,8+1)[rand_idx(8)]
    args['h'] = 500 #[50,100,200,500][rand_idx(4)]
    args['lr'] = 10**numpy.random.uniform(low=-5,high=-1)
    args['batch_size'] = 64#[32,64,128,256,512][rand_idx(5)]

    args['ID'] =  'hidden_layers_{}_h_{}_lr_{:.1e}_batch_size_{}'.format(
            args['hidden_layers'],args['h'],args['lr'],args['batch_size']) 

    print('[{}/{}] hidden layers: {}, h: {}, learning rate: {:.4e}, batch size: {}\n'.format(
        i,N,args['hidden_layers'],args['h'],args['lr'],args['batch_size'])) 
    sys.stdout.flush()

    # run training loop
    run.run(args)
