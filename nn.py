import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1 = torch.nn.Linear(4*3,200)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.fc2 = torch.nn.Linear(200,100)
        self.batchnorm2 = torch.nn.BatchNorm1d(100)
        self.fc3 = torch.nn.Linear(100,1)

    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = torch.nn.functional.dropout(x,training=self.training)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = torch.nn.functional.dropout(x,training=self.training)
        x = self.fc3(x)
        x = torch.nn.functional.log_softmax(x,dim=1)
        return x

