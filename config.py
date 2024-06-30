import torch


class Config:
    SHAKESPEARE_DATA = 'data/tinyshakespeare.txt'
    BATCH_SIZE = 32
    BLOCK_SIZE = 8
    MAX_ITERS = 5000
    EVAL_INTERVAL = 300
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_ITERS = 200
    N_EMBED = 32
    N_HEADS = 2
    N_LAYERS = 2
    DROPOUT = 0.2
    SEED = 1337