from ...config import SHAKESPEARE_DATA


def load_shakespeare_data():
    with open(SHAKESPEARE_DATA, 'r', encoding='utf-8') as f:
        return f.read()

def process_data(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return vocab_size, encode, decode