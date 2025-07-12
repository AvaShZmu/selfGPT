import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc

train_perc = 0.9
block_size = 8
batch_size = 32
iterations = 5000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200



with open('gpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabs = sorted(list(set(text)))
vocab_size = len(vocabs)

"""Making a basic encoder + decoder"""

decode_map = {i: char for i, char in enumerate(vocabs)}
encode_map = {char: i for i, char in enumerate(vocabs)}
encode = lambda s: [encode_map[char] for char in s]
decode = lambda l: ''.join(decode_map[i] for i in l)

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


xb, yb = get_batch('train')


class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Uses a lookup table
        self.token_embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.token_embed(idx)  # (Batch, Time, Channel)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)  # conform to Pytorch cross entropy
            target = target.reshape(B * T)
            # Logits: (B, T, C)
            # Target: (B, T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        idx = idx.to(device)
        for i in range(max_new_tokens):
            logits, _ = self(idx)  # Compute logits
            logits = logits[:, -1, :]  # Looks at the latest logit
            probs = F.softmax(logits, dim=-1)  # Computes the probability for next char
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = Bigram(vocab_size).to(device)
logits, loss = model(xb, yb)

# Generate
idx = torch.zeros((1, 1), dtype=torch.long)
print("First generation (without training):")
print("-"*25)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
print("-"*25)

"""Train the model"""

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(iterations):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

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
print("-"*25)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
print("-"*25)

gc.collect()
torch.cuda.empty_cache()