import subprocess
import numpy
import train
import load
import os

args = dict(
    h = 64,
    lr = 0.001,
    batch_size = 64,
    log_interval = 1000,
    epochs = 20
)

lrs = numpy.tile([1e-2,1e-3,1e-4,1e-5,1e-6],5)
batch_sizes = numpy.repeat([32,64,128,256,512],5)

for lr,batch_size in zip(lrs,batch_sizes):
    
    # update + print hyperparameters
    args['lr'] = lr; args['batch_size'] = int(batch_size)
    print('Learning rate: {:.1e}, batch size: {}, h: {}\n'.format(
        args['lr'],args['batch_size'],args['h']))
    sys.stdout.flush()

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    train_loss, val_loss, val_accuracy = evaluate_model(args,train_loader,val_loader)

    # save results
    s =  'lr_{:.1e}_batch_size_{}_h_{}'.format(args['lr'],args['batch_size'],args['h'])
    if s not in os.listdir('data/'):
        subprocess.call(f'mkdir data/{s}',shell=True)
    header = 'args = {}'.format(repr(args))
    numpy.savetxt(f'data/{s}/train_loss_{s}.dat',train_loss,header=header)
    numpy.savetxt(f'data/{s}/val_loss_{s}.dat',val_loss,header=header)
    numpy.savetxt(f'data/{s}/val_accuracy_{s}.dat',val_accuracy,header=header)
