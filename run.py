import subprocess
import numpy
import torch
import train
import load
import os

def run(args):

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    model, train_loss, val_loss, val_accuracy = train.evaluate_model(args,train_loader,val_loader)

    # save results
    if args['save']:
        ID = args['ID']
        if ID not in os.listdir('data/'):
            subprocess.call(f'mkdir data/results/{ID}',shell=True)
        header = 'args = {}'.format(repr(args))
        numpy.savetxt(f'data/results/{ID}/train_loss_{ID}.dat',train_loss,header=header)
        numpy.savetxt(f'data/results/{ID}/val_loss_{ID}.dat',val_loss,header=header)
        numpy.savetxt(f'data/results/{ID}/val_accuracy_{ID}.dat',val_accuracy,header=header)
        if not args['early_stopping']:
            torch.save(model.state_dict(),f'data/results/{ID}/model_{ID}.pt')
