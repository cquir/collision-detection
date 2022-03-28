import subprocess
import numpy
import torch
import wandb
import run
import sys
import os

args = dict(
    i = 2*3+2*4,
    dropout = False,
    early_stopping = False,
    test_batch_size = 1000,
    log_interval = 100,
    epochs = 100,
    patience = 30,
    seed = 0,
    save = True,
    validation = False,
    training_examples = 6.4*1e5
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

rand_idx = lambda ln: numpy.nonzero(numpy.random.multinomial(n=1,pvals=ln*[1./ln]))[0][0]

N = 100
for i in range(N):

    # update + print hyperparameters
    args['hidden_layers'] = numpy.arange(2,3+1)[rand_idx(2)]
    args['h'] = 500#[50,100,200,500][rand_idx(4)]
    args['lr'] = 10**numpy.random.uniform(low=-6,high=-3)
    args['batch_size'] = 64#[32,64,128,256,512][rand_idx(5)]

    wandb.init(config=args,project='collision-detection',reinit=True)

    args['ID'] =  'hidden_layers_{}_h_{}_lr_{:.1e}_batch_size_{}'.format(
            args['hidden_layers'],args['h'],args['lr'],args['batch_size']) 

    # run training loop
    run.run(args)
