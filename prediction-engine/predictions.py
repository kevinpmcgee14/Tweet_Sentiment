import sys 
sys.path.insert(1, '/opt')
import unzip_requirements
import torch
from install_compiled_dependencies import download_dependency
download_dependency('trained_twitter_model_half.pth')

def load_model(name):
    model = torch.load(name)
    model.eval()
    model.cpu()
    return model

model = load_model('trained_twitter_model_half.pth')

def handler(event, context):
    token = torch.tensor(event['token'])
    pred = model(token)
    pred = torch.argmax(pred).item()
    return {
        'status': 200,
        'prediction': pred
    }