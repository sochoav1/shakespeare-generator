import torch

from ..config import Config
from ..utils.data_loader import DataLoader
from .model import BigramLanguageModel


class Trainer:
    def __init__(self, config: Config, model: BigramLanguageModel, data_loader: DataLoader):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.EVAL_INTERVAL)
            for k in range(self.config.EVAL_INTERVAL):
                X, Y = self.data_loader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        for iter in range(self.config.MAX_ITERS):
            if iter % self.config.EVAL_INTERVAL == 0:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.data_loader.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def generate_text(self, max_new_tokens=500):
        context = torch.zeros((1, 1), dtype=torch.long, device=self.config.DEVICE)
        return self.data_loader.decode(self.model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.data_loader.vocab_size,
            'config': self.config.__dict__,
            'stoi': self.data_loader.stoi,
            'itos': self.data_loader.itos
        }, path)
        print(f"Modelo completo guardado en {path}")
