from calendar import EPOCH
import os
import torch
from torch.utils.data import DataLoader
from dataloader import ColliderDataset
# from addy_nn import NN
from nn import NeuralNetwork
import random

if not os.path.isdir("./data/checkpoints"):
    os.mkdir("./data/checkpoints")

learning_rate = 0.01
BATCH_SIZE = 2500
EPOCHS = 100_000

def main():
    train_dataset = ColliderDataset("data/datasets/test")
    data_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)

    use_cuda = torch.cuda.is_available()
    print("CUDA", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = NN().to(device)
    model = NeuralNetwork(14,128,5,False).to(device)

    if os.path.isfile('data/checkpoints/model.pt'):
        print("loading progress from file")
        model.load_state_dict(torch.load('data/checkpoints/model.pt'))
        model.eval()

    # model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.LogSoftmax()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    torch.cuda.manual_seed_all(5)

    for epoch in range(EPOCHS):
        loss = 0
        correct = 0
        total = 0

        model.train()

        for batch_idx, (x, y) in enumerate(data_loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model.forward(x)

            loss = criterion(y_pred,y.unsqueeze(1))
            loss.backward()

            optimizer.step()
            if epoch % 2 == 0:
                with torch.no_grad():
                    correct += (torch.round(torch.sigmoid(y_pred).squeeze(1)) == y).sum().item()

                total += 1

        if epoch % 2 == 0:
            torch.save(model.state_dict(),'data/checkpoints/model.pt')
        
            total = total * BATCH_SIZE
            print(correct/total)




if __name__ == "__main__":
    main()
