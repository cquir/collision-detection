import subprocess
import numpy
import torch
import train
import load
import os

def run(args):
	bdir = f'../data/results/{args["name"]}'
	if not os.path.isdir(bdir):
		os.mkdir(bdir)
	# save run id to resume wandb run 
	numpy.savetxt(bdir+f'/{args["name"]}-run-id.txt',[args['run_id']],fmt='%s')
	# load data + train model
	train_loader,val_loader = load.load_data(args)
	train.evaluate_model(args,train_loader,val_loader)
