import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shakespeare_ml.config import Config
from shakespeare_ml.model import BigramLanguageModel
from shakespeare_ml.utils.data_loader import DataLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las fuentes. Puedes restringir esto a tu frontend.
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Cargar el modelo entrenado
config = Config()
model_path = "shakespeare_ml/trained_models/modelo_lenguaje_completo.pt"
checkpoint = torch.load(model_path)

data_loader = DataLoader(config)
data_loader.stoi = checkpoint['stoi']
data_loader.itos = checkpoint['itos']

model = BigramLanguageModel(config, len(data_loader.stoi)).to(config.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class GenerationRequest(BaseModel):
    max_tokens: int = 100

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
        generated = model.generate(context, max_new_tokens=request.max_tokens)
        text = data_loader.decode(generated[0].tolist())
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)