import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):  # A single self-attention head
    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Computes attention score (affinity)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the vals
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHead(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(n_embd, head_size, dropout, block_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # A transformer block
    def __init__(self, n_embd, n_heads, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHead(n_embd, n_heads, head_size, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Uses layernorm + residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class BasicGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        # Uses a lookup table
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_head, dropout=dropout, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_emb = self.token_embed(idx)  # (Batch, Time, Channel=n_embd)
        pos_emb = self.pos_embed(torch.arange(T, device=self.device)).unsqueeze(0)  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  #(B, T, vocab_size)

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
        idx = idx.to(self.device)
        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)  # Compute logits
            logits = logits[:, -1, :]  # Looks at the latest logit
            probs = F.softmax(logits, dim=-1)  # Computes the probability for next char
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
