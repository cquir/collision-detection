import numpy
import torch
import sys
import nn
import os

def evaluate_model(args,train_loader,val_loader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = nn.NeuralNetwork(args['i'],args['h'],args['hidden_layers'],args['dropout']).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=args['momentum'])
    optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'])

    # train the neural network 
    def train(epoch):
        model.train()  
        for batch_idx, (data,label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
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

    # evaluate the neural network's performance
    def test(test_loader,dataset_label,batch_size):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data,label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                test_loss += criterion(output,label.unsqueeze(1)).item()
                # sigmoid layer normally included in criterion
                pred = torch.round(torch.sigmoid(output))
                correct += (pred.squeeze(1) == label).sum().item()
        test_loss *= batch_size/len(test_loader.dataset) 
        print('{} set: Average loss: {:.4f}, Accuracy: {:.1f}%'.format(
            dataset_label,test_loss,100.*correct/len(test_loader.dataset)))
        sys.stdout.flush()
        return test_loss, correct/len(test_loader.dataset)

    train_losses = []; val_losses = [] 
    val_accuracies = []; train_accuracies = []
    epochs = range(1,args['epochs']+1)
    best_score = None; early_stop = False; counter = 0

    test(train_loader,'Training',args['batch_size'])
    if args['validation']:
        test(val_loader,'Validation',args['test_batch_size'])
    print(''); sys.stdout.flush()

    for epoch in epochs:

        train(epoch)
        print(''); sys.stdout.flush()
        train_loss, train_accuracy = test(train_loader,'Training',args['batch_size'])
        train_losses.append(train_loss); train_accuracies.append(train_accuracy)
        if args['validation']:
            val_loss, val_accuracy = test(val_loader,'Validation',args['test_batch_size'])
            val_losses.append(val_loss); val_accuracies.append(val_accuracy)
        print(''); sys.stdout.flush()

        # early stopping
        if args['early_stopping']:
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
                if args['save']:
                    ID = args['ID']
                    torch.save(model.state_dict(),'data/results/{ID}/model_{ID}.pt')
                counter = 0
            if early_stop:
                print(f'Early Stopping!\n')
                sys.stdout.flush()
                break
        
    return model, train_losses, val_losses, train_accuracies, val_accuracies
