import torch

from ..config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.chars = None
        self.vocab_size = None
        self.stoi = None
        self.itos = None
        self.train_data = None
        self.val_data = None

    def load_and_process_data(self):
        with open(self.config.SHAKESPEARE_DATA, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.BLOCK_SIZE, (self.config.BATCH_SIZE,))
        x = torch.stack([data[i:i+self.config.BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+self.config.BLOCK_SIZE+1] for i in ix])
        return x.to(self.config.DEVICE), y.to(self.config.DEVICE)