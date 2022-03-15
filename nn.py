import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self,h):
        super(NeuralNetwork,self).__init__()
        self.fc1 = torch.nn.Linear(4*3,h)
        self.batchnorm1 = torch.nn.BatchNorm1d(h)
        self.fc2 = torch.nn.Linear(h,h)
        self.batchnorm2 = torch.nn.BatchNorm1d(h,h)
        self.fc3 = torch.nn.Linear(h,1)

    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = torch.nn.functional.dropout(x,training=self.training)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = torch.nn.functional.dropout(x,training=self.training)
        x = self.fc3(x)
        return x
