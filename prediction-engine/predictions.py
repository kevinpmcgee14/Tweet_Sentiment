import torch
import torchtext
from tokenizer import Tokenizer
from model import model
from install_compiled_dependencies import download_dependency

tokenizer_name = 'tweet_tokenizer.pkl'
model_weights = 'model_best_weights.pth'

download_dependency(model_weights)
download_dependency(tokenizer_name)

def load_tokenizer(tok_name):
    tok = pickle.load('/tmp/' + tok_name)
    return tok

def load_model(name):
    model.load_state_dict(name, strict=False)
    model.eval()
    model.cpu()
    return model

tokenizer = load_tokenizer(tokenizer_name)
model = load_model(model_weights)

def main(event, context):
    if not event.get('text'):
        return "Hello from Lambda!"
    token = torch.tensor(tokenizer(event['text']).unsqueeze(0))
    pred = model(token)
    pred = torch.argmax(pred).item()
    return {
        'status': 200,
        'prediction': pred
    }