import torch
import torch.nn as nn
from gpt_model import BasicGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'shakespeare-basicgpt-val-1.52.pth'
config = torch.load(f'../trained_models/{model_name}', map_location=device)


vocabs = config['vocabs']
itos = {i: char for i, char in enumerate(vocabs)}
stoi = {char: i for i, char in enumerate(vocabs)}


def encode(text):
    unk_idx = stoi['<unk>']
    return [stoi.get(ch, unk_idx) for ch in text]


def decode(indices):
    return ''.join(itos[i] for i in indices)


model_args = config['model_args']
model = BasicGPT(**model_args, device=device).to(device)
model.load_state_dict(config['model_state_dict'])
model.eval()

text_inp = input("Input text: ")
encoded = torch.unsqueeze(torch.tensor(encode(text_inp)), 0)
answer = decode(model.generate(encoded, max_new_tokens=1000)[0].tolist())
print(answer)
