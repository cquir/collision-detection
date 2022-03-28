import subprocess
import numpy
import torch
import train
import load
import os

def run(args):

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    model, train_losses, val_losses, train_accuracies, val_accuracies = train.evaluate_model(args,train_loader,val_loader)

    # save results
    if args['save']:
        ID = args['ID']
        if ID not in os.listdir('../data/results/'):
            subprocess.call(f'mkdir ../data/results/{ID}',shell=True)
        header = 'args = {}'.format(repr(args))
        numpy.savetxt(f'../data/results/{ID}/train_loss_{ID}.dat',train_losses,header=header)
        numpy.savetxt(f'../data/results/{ID}/train_accuracy_{ID}.dat',train_accuracies,header=header)
        if args['validation']:
            numpy.savetxt(f'../data/results/{ID}/val_loss_{ID}.dat',val_losses,header=header)
            numpy.savetxt(f'../data/results/{ID}/val_accuracy_{ID}.dat',val_accuracies,header=header)
        if not args['early_stopping']:
            torch.save(model.state_dict(),f'../data/results/{ID}/model_{ID}.pt')
