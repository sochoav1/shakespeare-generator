import os

import torch

# Rutas de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'shakespeare_ml', 'data')
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'shakespeare_ml', 'trained_models')

# Archivo de datos
SHAKESPEARE_DATA = os.path.join(DATA_DIR, 'tinyshakespeare.txt')

# Ruta para guardar/cargar el modelo entrenado
MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'shakespeare_model.pth')

# Configuración del modelo
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

# Configuración de la API
API_HOST = "0.0.0.0"
API_PORT = 8000