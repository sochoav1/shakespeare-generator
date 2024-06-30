import os

import torch

from ..config import Config
from ..model import BigramLanguageModel, Trainer
from ..utils.data_loader import DataLoader


def main():
    config = Config()
    torch.manual_seed(config.SEED)

    data_loader = DataLoader(config)
    data_loader.load_and_process_data()

    model = BigramLanguageModel(config, data_loader.vocab_size).to(config.DEVICE)

    trainer = Trainer(config, model, data_loader)
    trainer.train()

    generated_text = trainer.generate_text()
    print(generated_text)

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models'))
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'dumb-shakespeare.pt')
    trainer.save_model(save_path)

if __name__ == "__main__":
    main()