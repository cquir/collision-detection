import torch
import nn

# load pytorch model
name = 'earthy-durian-312'
model = nn.NeuralNetwork(i=7,h=500,hidden_layers=6,dropout=False,pdrop=0)
checkpoint = torch.load(f'../data/results/{name}/model-{name}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# convert and save pytorch model to onnx model
dummy_input = torch.zeros(1,7)
torch.onnx.export(model,dummy_input,f'../data/results/{name}/model-{name}.onnx',verbose=True)

