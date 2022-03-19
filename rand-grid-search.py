import subprocess
import numpy
import train
import load
import sys
import os

args = dict(
    h = 64,
    lr = 0.001,
    momentum = 0.5,
    batch_size = 64,
    test_batch_size = 1000,
    log_interval = 1000,
    epochs = 100,
    patience = 10
)

for i in range(50):
    # update + print hyperparameters
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=4*[0.25]))[0][0]
    args['h'] = [50,100,200,500][idx]
    args['lr'] = 10**numpy.random.uniform(low=-5,high=-1)
    args['momentum'] = numpy.random.uniform()
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=4*[0.25]))[0][0]
    args['batch_size'] = [32,64,128,256,512][idx]
    print('h: {}, learning rate: {:.4e}, momentum: {:.5f}, batch size: {}\n'.format(
        args['h'],args['lr'],args['momentum'],args['batch_size'])) 
    sys.stdout.flush()

    # load data + train model
    train_loader,val_loader = load.load_data(args)
    model, train_loss, val_loss, val_accuracy = train.evaluate_model(args,train_loader,val_loader)

    # save results
    s =  'h_{}_lr_{:.1e}_momentum_{:.5f}_batch_size_{}'.format(
        args['h'],args['lr'],args['momentum'],args['batch_size']) 
    if s not in os.listdir('data/'):
        subprocess.call(f'mkdir data/{s}',shell=True)
    header = 'args = {}'.format(repr(args))
    numpy.savetxt(f'data/{s}/train_loss_{s}.dat',train_loss,header=header)
    numpy.savetxt(f'data/{s}/val_loss_{s}.dat',val_loss,header=header)
    numpy.savetxt(f'data/{s}/val_accuracy_{s}.dat',val_accuracy,header=header)
    torch.save(model.state_dict(),f'data/{s}/model_{s}.pt')
