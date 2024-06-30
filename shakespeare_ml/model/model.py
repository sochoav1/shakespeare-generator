import torch
import torch.nn as nn
from torch.nn import functional as F

from shakespeare_ml.config import Config, config


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.N_EMBED, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBED, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.N_EMBED, 4 * config.N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * config.N_EMBED, config.N_EMBED),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        head_size = config.N_EMBED // config.N_HEADS
        self.sa = MultiHeadAttention(config, config.N_HEADS, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.N_EMBED)
        self.ln2 = nn.LayerNorm(config.N_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, config: Config, vocab_size):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBED)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.N_LAYERS)])
        self.ln_f = nn.LayerNorm(config.N_EMBED)
        self.lm_head = nn.Linear(config.N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=self.config.DEVICE))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.config.BLOCK_SIZE:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
