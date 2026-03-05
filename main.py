from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import io

import time
import torch.nn.functional as F

app = FastAPI()

# -------------------------------
# CONFIG
# -------------------------------

MODEL_PATH = "model/best_model.pth"
NUM_CLASSES = 6

CLASS_NAMES = [
    "Bouteille_plastique",
    "Brique_en_carton",
    "Emballage_metallique",
    "Ordure_ménagère",
    "Papier_Carton",
    "Verre"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD MODEL (Vision Mamba)
# -------------------------------

model = timm.create_model(
    "mambaout_tiny",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------------------
# TRANSFORM
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# ROUTES
# -------------------------------

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import torch.nn.functional as F


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    inference_time = round((time.time() - start_time) * 1000, 2)

    result = class_names[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    probs_dict = {
        class_names[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(len(class_names))
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": {},
            "result": result,
            "confidence": confidence_score,
            "probabilities": probs_dict,
            "inference_time": inference_time,
        },
    )
