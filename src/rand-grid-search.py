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
	seed = 1,
	dataset_examples = 3e8,
	dropout = False,
	pdrop = 0.0,
	early_stopping = True,
	threshold = 1.0,
	split = 1.0,
	pct_start = 0.3,
	max_momentum = 0.97,
	hidden_layers = 10,
	h = 500,
	batch_size = 256,
	lr = 1e-4,
	resume = False,
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

if args['resume']:
	args['name'] = 'swept-star-343'
	# resume run to track with wandb
	args['run_id'] = str(numpy.loadtxt(f'../data/results/{args["name"]}/{args["name"]}-run-id.txt',dtype=str))
	obj = wandb.init(config=args,project='collision-detection',id=args['run_id'],resume='allow')
else:
	# initialize new run to track with wandb
	args['run_id'] = wandb.util.generate_id()
	obj = wandb.init(config=args,project='collision-detection',id=args['run_id'])
	args['name']  = obj.name

# run training loop
run.run(args)
