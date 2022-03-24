import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self,i,h,hidden_layers,dropout):
        super(NeuralNetwork,self).__init__()
        self.dropout = dropout
        self.fc1 = torch.nn.Linear(i,h)
        self.batchnorm1 = torch.nn.BatchNorm1d(h)
        self.layers = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList()
        for i in range(hidden_layers):
            self.layers.append(torch.nn.Linear(h,h))
            self.batchnorms.append(torch.nn.BatchNorm1d(h))
        self.layer_out = torch.nn.Linear(h,1)

    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.batchnorm1(x)
        if self.dropout:
            x = torch.nn.functional.dropout(x,training=self.training)
        for layer,batchnorm in zip(self.layers,self.batchnorms):
            x = torch.nn.functional.relu(layer(x))
            x = batchnorm(x)
            if self.dropout:
                x = torch.nn.functional.dropout(x,training=self.training)
        x = self.layer_out(x)
        return x
