import torch
import numpy
import train
import wandb
import click
import load
import os

args = dict(
	i = 3+4,
	test_batch_size = 1000,
	epochs = 100, 
	patience = 10,
	seed = 1,
	dataset_examples = 1e7,
	dropout = False,
	pdrop = 0.0,
	early_stopping = False,
	threshold = 1.0,
	split = 1.0,
	pct_start = 0.3,
	max_momentum = 0.95,
	hidden_layers = 6,
	h = 500,
	batch_size = 256,
	lr = 1e-4,
)

@click.command()
@click.option('--resume','resume',default=False,help='resume run (boolean)')
@click.option('--name','name',default=None,help='name of model')
def run(resume,name):
	torch.manual_seed(args['seed'])
	numpy.random.seed(args['seed'])
	if resume:
		# resume run to track with wandb
		args['run_id'] = str(numpy.loadtxt(f'../data/results/{args["name"]}/{args["name"]}-run-id.txt',dtype=str))
		obj = wandb.init(config=args,project='collision-detection',id=args['run_id'],resume='allow')
	else:
		# initialize new run to track with wandb
		args['run_id'] = wandb.util.generate_id()
		obj = wandb.init(config=args,project='collision-detection',id=args['run_id'])
		args['name']  = obj.name
		bdir = f'../data/results/{args["name"]}'
		if not os.path.isdir(bdir):
			os.mkdir(bdir)
		numpy.savetxt(bdir+f'/{args["name"]}-run-id.txt',[args['run_id']],fmt='%s')
	# load data + train model
	train_loader,val_loader = load.load_data(args)
	train.evaluate_model(args,train_loader,val_loader)

if __name__ == "__main__":
	run()