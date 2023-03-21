import onnxruntime
import numpy
import torch
import onnx
import nn

device = torch.device('cpu') ## REMOVE LATER

# load pytorch model
name = 'sleek-breeze-268'
model = nn.NeuralNetwork(i=7,h=500,hidden_layers=6,dropout=False,pdrop=0).to(device)
checkpoint = torch.load(f'../data/results/model-{name}.pt',map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# convert and save pytorch model to onnx model
dummy_input = torch.zeros(1,7)
torch.onnx.export(model,dummy_input,f'../data/results/model-{name}.onnx',input_names=['input'],output_names=['output'])