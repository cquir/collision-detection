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
    epochs = 200, 
    patience = 10,
    seed = 1,
    validation = True,
    dataset_examples = 5e7,
    dropout = False,
    pdrop = 0.0,
    early_stopping = False,
    threshold = 1.0,
	split = 1.0,
	pct_start = 0.3,
	max_momentum = 0.97,
	hidden_layers = 6,
	h = 500,
	batch_size = 256,
	lr = 1e-4
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

obj = wandb.init(config=args,project='collision-detection',reinit=True)
args['name']  = obj.name

# run training loop
run.run(args)
