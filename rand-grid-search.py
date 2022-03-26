import numpy
import torch
import run
import sys

args = dict(
    i = 2*3+2*4,
    dropout = False,
    early_stopping = False,
    test_batch_size = 1000,
    log_interval = 1000,
    epochs = 200,
    patience = 30,
    seed = 0,
    save = False
)

torch.manual_seed(args['seed'])
numpy.random.seed(args['seed'])

N = 100
for i in range(N):

    # update + print hyperparameters
    args['hidden_layers'] = int(sys.argv[1])
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=4*[0.25]))[0][0]
    args['h'] = [50,100,200,500][idx]
    args['lr'] = 10**numpy.random.uniform(low=-5,high=-1)
    idx = numpy.nonzero(numpy.random.multinomial(n=1,pvals=5*[0.2]))[0][0]
    args['batch_size'] = [32,64,128,256,512][idx]
    args['ID'] =  'hidden_layers_{}_h_{}_lr_{:.1e}_batch_size_{}'.format(
            args['hidden_layers'],args['h'],args['lr'],args['batch_size']) 

    print('[{}/{}] hidden layers: {}, h: {}, learning rate: {:.4e}, batch size: {}\n'.format(
        i,N,args['hidden_layers'],args['h'],args['lr'],args['batch_size'])) 
    sys.stdout.flush()

    # run training loop
    run.run(args)
