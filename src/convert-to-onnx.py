import torch
import click
import nn

@click.command()
@click.argument('name')
def convert(name):
    device = torch.device('cpu') 
    # load pytorch model
    model = nn.NeuralNetwork(i=7,h=500,hidden_layers=10,dropout=False,pdrop=0).to(device)
    checkpoint = torch.load(f'../data/results/{name}/model-{name}.pt',map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # convert and save pytorch model to onnx model
    dummy_input = torch.zeros(1,7)
    torch.onnx.export(
        model,
        dummy_input,
        f'../data/results/{name}/model-{name}.onnx',
        input_names=['input'],
        output_names=['output']
    )

if __name__ == "__main__":
    convert()