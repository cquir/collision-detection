import subprocess
import numpy
import torch
import train
import load
import os

def run(args):

    name = args['name']
    if not os.path.isdir(f'../data/results/{name}'):
        os.mkdir(f'../data/results/{name}')

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    model, train_losses, val_losses, train_accuracies, val_accuracies = train.evaluate_model(args,train_loader,val_loader)

    # save results
    header = 'args = {}'.format(repr(args))
    numpy.savetxt(f'../data/results/{name}/train-loss-{name}.dat',train_losses,header=header)
    numpy.savetxt(f'../data/results/{name}/train-accuracy-{name}.dat',train_accuracies,header=header)
    if args['validation']:
        numpy.savetxt(f'../data/results/{name}/val-loss-{name}.dat',val_losses,header=header)
        numpy.savetxt(f'../data/results/{name}/val-accuracy-{name}.dat',val_accuracies,header=header)
    if not args['early_stopping']:
        torch.save(model.state_dict(),f'../data/results/{name}/model-{name}.pt')
