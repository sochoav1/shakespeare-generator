import torch


class Config:
    SHAKESPEARE_DATA = 'data/tinyshakespeare.txt'
    BATCH_SIZE = 1
    BLOCK_SIZE = 1  # Match this with the checkpoint
    MAX_ITERS = 5000
    EVAL_INTERVAL = 500
    LEARNING_RATE = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_ITERS = 200
    N_EMBED = 1  # Match this with the checkpoint
    N_HEADS = 1    # Match this with the checkpoint
    N_LAYERS = 1  # Match this with the checkpoint
    DROPOUT = 0.2
    SEED = 1337
    
