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

def generate_text(prompt, max_tokens=500, temperature=0.8, top_k=50):
    encoded = torch.tensor(encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = model.generate(
            encoded,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    return decode(generated[0].tolist())

# Interactive loop
while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ")
    if prompt.lower() == 'quit':
        break

    temp = float(input("Temperature (0.1-2.0, default 0.8): ") or "0.8")
    max_tokens = int(input("Max tokens (default 200): ") or "200")

    result = generate_text(prompt, max_tokens, temp)
    print(f"\nGenerated text:\n{result}")
