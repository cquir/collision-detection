import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self ,i, h, hidden_layers, dropout):
        super(NeuralNetwork,self).__init__()
        self.dropout = dropout 
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers+1):
            self.layers.append(torch.nn.Linear(i,h))
            self.layers.append(torch.nn.BatchNorm1d(h))
            i = h
        self.layers.append(torch.nn.Linear(h,1))

    def forward(self,x):
        for n in range(int(len(self.layers[:-1])/2)):
            layer = self.layers[2*n]; batchnorm = self.layers[2*n+1] 
            x = torch.nn.functional.relu(layer(x))
            x = batchnorm(x)
            if self.dropout:
                x = torch.nn.functional.dropout(x,training=self.training)
        return self.layers[-1](x)