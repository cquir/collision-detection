import numpy
import torch
import sys
import nn
import os

def evaluate_model(args,train_loader,val_loader):

    model = nn.NeuralNetwork(args['h']) 
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=args['momentum'])
    #optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'])

    # train the neural network 
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
                sys.stdout.flush()
        return loss.item()

    # evaluate the neural network's performance
    def test(test_loader,dataset_label):
        model.eval()
        test_loss = 0
        correct = 0
        for data,label in test_loader:
            output = model(data)
            test_loss += criterion(output,label.unsqueeze(1)).item()
            # sigmoid layer normally included in criterion
            pred = torch.round(torch.sigmoid(output))
            correct += (pred.squeeze(1) == label).sum().item()
        test_loss *= args['test_batch_size']/len(test_loader.dataset) 
        print('\n{} set: Average loss: {:.4f}, Accuracy: {:.1f}%'.format(
            dataset_label,test_loss,100.*correct/len(test_loader.dataset)))
        sys.stdout.flush()
        return test_loss, correct/len(test_loader.dataset)

    train_loss = numpy.zeros((args['epochs'],))
    val_loss = numpy.zeros_like(train_loss)
    val_accuracy = numpy.zeros_like(train_loss)
    epochs = range(1,args['epochs']+1)
    best_score = None; early_stop = False

    for epoch in epochs:

        train_loss[epoch-1] = train(epoch)
        val_loss[epoch-1], val_accuracy[epoch-1] = test(val_loader,'Validation')

        # early stopping
        if best_score is None:
            best_score = val_loss[epoch-1]
        elif val_loss[epoch-1] >= best_score:
            counter += 1
            print(f'Early stopping counter: {counter}\n')
            sys.stdout.flush()
            if counter >= args['patience']:
                early_stop = True
        else:
            best_score = val_loss[epoch-1]
            counter = 0
        if early_stop:
            print('Early Stopping')
            sys.stdout.flush()
            break
        
    return model, train_loss, val_loss, val_accuracy
