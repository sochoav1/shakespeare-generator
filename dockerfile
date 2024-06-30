# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de requerimientos
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente del proyecto al contenedor
COPY . .

# Crear el directorio para guardar el modelo
RUN mkdir -p /app/shakespeare_ml/trained_models

# Instalar el proyecto
RUN pip install -e .

# Comando para ejecutar el script de entrenamiento
CMD ["python", "-m", "shakespeare_ml.scripts.train_model"]
