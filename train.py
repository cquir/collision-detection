import numpy
import torch
import load
import nn

args = dict(
    h = 16,
    lr = 0.001,
    batch_size = 64,
    log_interval = 1000,
    epochs = 30
)

#lrs = numpy.tile([1e-1,1e-2,1e-3,1e-4,1e-5],5)
#hs = numpy.repeat([16,32,64,128,256],5)

lrs = [1e-2,1e-3,1e-4,1e-5]
h = 256

#for lr,h in zip(lrs,hs):
for lr in lrs:
    
    print('Learning rate: {:.1e}, h: {}\n'.format(lr,h))
    args['lr'] = lr; args['h'] = h

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
            pred = torch.round(torch.sigmoid(output))
            correct = (pred.squeeze(1) == label).sum().item()
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.1f}%'.format(
                    epoch, batch_idx*len(data),len(train_loader.dataset),
                    100.*batch_idx/len(train_loader),loss.item(),100.*correct/len(data)))
        return loss.item()

    def test(test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data,label in test_loader:
            output = model(data)
            test_loss += criterion(output,label.unsqueeze(1)).item()
            pred = torch.round(torch.sigmoid(output))
            correct += (pred.squeeze(1) == label).sum().item()
        test_loss *= args['batch_size']/len(test_loader.dataset) 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}%\n'.format(
            test_loss,100.*correct/len(test_loader.dataset)))
        return test_loss, correct/len(test_loader.dataset)

    train_loss = numpy.zeros((args['epochs'],))
    val_loss = numpy.zeros_like(train_loss)
    val_accuracy = numpy.zeros_like(train_loss)
    epochs = range(1,args['epochs']+1)
    for epoch in epochs:
        train_loss[epoch-1] = train(epoch)
        val_loss[epoch-1], val_accuracy[epoch-1] = test(val_loader)
    
    s =  'lr_{:.1e}_h_{}'.format(args['lr'],args['h'])
    numpy.savetxt('data/train_loss_{}.dat'.format(s),train_loss)
    numpy.savetxt('data/val_loss_{}.dat'.format(s),val_loss)
    numpy.savetxt('data/val_accuracy_{}.dat'.format(s),val_accuracy)
