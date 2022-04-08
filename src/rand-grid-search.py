import subprocess
import numpy
import torch
import wandb
import run
import sys
import os

args = dict(
    i = 3+4,
    test_batch_size = 1000,
    epochs = 100, 
    patience = 10,
    seed = 0,
    validation = True,
    dataset_examples = 1e6,
    dropout = False,
    pdrop = 0.0,
    early_stopping = False,
    threshold = 1.0
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

rand_idx = lambda ln: numpy.nonzero(numpy.random.multinomial(n=1,pvals=ln*[1./ln]))[0][0]

N = 1
for idx in range(N):
    # update  hyperparameters
    args['hidden_layers'] = 5
    args['h'] = 100
    args['lr'] = 0.0005
    args['batch_size'] = 256

    obj = wandb.init(config=args,project='collision-detection',reinit=True)
    args['name']  = obj.name

    # run training loop
    run.run(args)
