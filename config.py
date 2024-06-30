import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración de rutas
DATA_DIR = os.path.join(BASE_DIR, 'model', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Archivo de datos
SHAKESPEARE_DATA = os.path.join(DATA_DIR, 'tinyshakespeare.txt')

# Otras configuraciones
MAX_TOKENS = 100
