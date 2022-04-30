import numpy
import torch
import wandb
import nn
import os

def evaluate_model(args,train_loader,val_loader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = nn.NeuralNetwork(args['i'],args['h'],args['hidden_layers'],args['dropout'],args['pdrop']).to(device)
    wandb.watch(model,log="all")
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=args['momentum'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args['lr'],steps_per_epoch=len(train_loader),epochs=args['epochs'],pct_start=args['pct_start'],max_momentum=args['max_momentum'])

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
            scheduler.step()
            
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
        return test_loss, correct/len(test_loader.dataset)

    train_losses = []; val_losses = [] 
    val_accuracies = []; train_accuracies = []
    epochs = range(1,args['epochs']+1)
    best_score = None; early_stop = False; counter = 0

    for epoch in epochs:

        train(epoch)
        train_loss, train_accuracy = test(train_loader,'Training',args['batch_size'])
        train_losses.append(train_loss); train_accuracies.append(train_accuracy)
        ldic = {'Train loss':train_loss,'Train accuracy':train_accuracy}
        if args['validation']:
            val_loss, val_accuracy = test(val_loader,'Validation',args['test_batch_size'])
            val_losses.append(val_loss); val_accuracies.append(val_accuracy)
            ldic['Validation loss'] = val_loss; ldic['Validation accuracy'] = val_accuracy
        wandb.log(ldic)

        # early stopping
        if args['early_stopping']:
            if best_score is None:
                best_score = val_loss
            elif val_loss >= best_score:
                counter += 1
                if counter >= args['patience']:
                    early_stop = True
            else:
                best_score = val_loss
                name = args['name']
                torch.save(model.state_dict(),f'../data/results/{name}/model-{name}.pt')
                counter = 0
            if early_stop:
                break

        # abort if training loss is above threshold after first epoch
        if train_loss >= args['threshold']:
            break
        
    return model, train_losses, val_losses, train_accuracies, val_accuracies
