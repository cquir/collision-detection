import torch
import nn

device = torch.device('cpu')

name = 'sleek-breeze-268'
model = nn.NeuralNetwork(i=7,h=500,hidden_layers=6,dropout=False,pdrop=0).to(device)
model.load_state_dict(torch.load(f'../model-{name}.pt',map_location=device))
model.eval()
output = model(torch.rand(1,7))
pred = torch.sigmoid(output)
print(output.detach().numpy()[0][0],pred.detach().numpy()[0][0])
