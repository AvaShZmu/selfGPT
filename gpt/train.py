import torch
import torch.nn as nn
from torch.nn import functional as F
import gc
from gpt_model import BasicGPT
from tqdm import tqdm, trange

# model args
n_embd = 192
n_head = 4
n_layer = 4
block_size = 256
dropout = 0.2

# other hyperparameters
train_perc = 0.9
batch_size = 32
iterations = 4000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For continued training
CHECKPOINT: str = None  # (Insert file name with .pth extension)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabs = sorted(list(set(text)))
vocabs.append('<unk>')
vocab_size = len(vocabs)

"""Making a basic encoder + decoder"""

itos = {i: char for i, char in enumerate(vocabs)}
stoi = {char: i for i, char in enumerate(vocabs)}

def encode(text):
    unk_idx = stoi['<unk>']
    return [stoi.get(ch, unk_idx) for ch in text]


def decode(indices):
    return ''.join(itos[i] for i in indices)


"""Encode the whole text input"""

data = torch.tensor(encode(text))

"""Split into train + test"""

n = len(data)
indice = int(n * train_perc)
train_data = data[:indice]
test_data = data[indice:]

"""Create batches of dataset"""

def get_batch(split):
    data = train_data if split == 'train' else test_data
    x_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in x_indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in x_indices])
    x, y = x.to(device), y.to(device)
    return x, y


"""A function for averaging losses over multiple batches"""

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""Check for checkpoints (For retraining purposes)"""
if CHECKPOINT:
    config = torch.load(f'../trained_models/{CHECKPOINT}', map_location=device)
    model_args = config['model_args']
    model = BasicGPT(**model_args, device=device)
    model.load_state_dict(config['model_state_dict'])

else:
    model = BasicGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)

model.to(device)

"""Train the model"""

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
progress = trange(1, iterations + 1, desc='Iterations')

final_val = 0
for iter in progress:
    if iter % eval_interval == 0 or iter == 1:
        losses = estimate_loss()
        progress.set_description(f"step {iter} | train {losses['train']:.3f} | val {losses['test']:.3f}")

    if iter == iterations:
        losses = estimate_loss()
        final_val = losses['test']
    # sample a batch
    xb, yb = get_batch('train')

    # eval loss
    logits, loss = model(xb, yb)  # Perform a forward pass
    optimizer.zero_grad(set_to_none=True)  # Zeroes out the gradient from prev
    loss.backward()  # Computing the gradient
    optimizer.step()  # Update the parameters using the optimizer

"""Try out the new model (obv not gonna be good)"""

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Now with some training:")
print("-" * 25)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
print("-" * 25)

torch.save({
    "model_args":{
    "vocab_size": vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "dropout": dropout,
    },
    "vocabs": vocabs,
    "model_state_dict": model.state_dict(),
}, f"../trained_models/tinystories-basicgpt-val-{final_val:.2f}.pth")

gc.collect()
torch.cuda.empty_cache()
