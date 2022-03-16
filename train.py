import subprocess
import numpy
import torch
import load
import nn
import os

args = dict(
    h = 64,
    lr = 0.001,
    batch_size = 64,
    log_interval = 1000,
    epochs = 20
)

lrs = numpy.tile([1e-2,1e-3,1e-4,1e-5,1e-6],5)
batch_sizes = numpy.repeat([],5)
hs = numpy.repeat([16,32,64,128,256],5)

for lr,batch_size in zip(lrs,batch_sizes):
    
    args['lr'] = lr; args['batch_size'] = batch_size

    print('Learning rate: {:.1e}, batch size: {}, h: {}\n'.format(
        args['lr'],args['batch_size'],args['h']))

    train_loader,val_loader = load.load_data(args)

    model = nn.NeuralNetwork(args['h'])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'])

    def train(epoch):
        model.train()
        for batch_idx, (data,label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx*len(data),len(train_loader.dataset),
                    100.*batch_idx/len(train_loader)))

    def test(test_loader,dataset_label):
        model.eval()
        test_loss = 0
        correct = 0
        for data,label in test_loader:
            output = model(data)
            test_loss += criterion(output,label.unsqueeze(1)).item()
            pred = torch.round(torch.sigmoid(output))
            correct += (pred.squeeze(1) == label).sum().item()
        test_loss *= args['batch_size']/len(test_loader.dataset) 
        print('{} set: Average loss: {:.4f}, Accuracy: {:.1f}%'.format(
            dataset_label,test_loss,100.*correct/len(test_loader.dataset)))
        return test_loss, correct/len(test_loader.dataset)

    train_loss = numpy.zeros((args['epochs'],))
    train_accuracy = numpy.zeros_like(train_loss)
    val_loss = numpy.zeros_like(train_loss)
    val_accuracy = numpy.zeros_like(train_loss)
    epochs = range(1,args['epochs']+1)
    for epoch in epochs:
        train(epoch)
        print('\n')
        train_loss[epoch-1], train_accuracy[epoch-1] = test(train_loader,'Train')
        val_loss[epoch-1], val_accuracy[epoch-1] = test(val_loader,'Validation')
        print('\n')
    
    # save data
    s =  'lr_{:.1e}_batch_size_{}_h_{}'.format(args['lr'],args['batch_size'],args['h'])
    if s not in os.listdir('data/'):
        subprocess.call(f'mkdir data/{s}',shell=True)
    header = 'args = {}'.format(repr(args))
    numpy.savetxt(f'data/{s}/train_loss_{s}.dat',train_loss)
    numpy.savetxt(f'data/{s}/val_loss_{s}.dat',val_loss)
    numpy.savetxt(f'data/{s}/val_accuracy_{s}.dat',val_accuracy)
