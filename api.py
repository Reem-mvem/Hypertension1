"""Minimal FastAPI inference API for the RETFound hypertension classifier."""

import io
import os
import sys
from typing import List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from timm.models.layers import trunc_normal_
from torchvision import transforms

# ==== USER PATHS (update these if your layout differs) ====
RETFOUND_ROOT = r"C:\Users\reem2\Downloads\train_eye\RETFound"
UPSTREAM_WEIGHTS_PATH = r"C:\Users\reem2\Downloads\train_eye\RETFound_cfp_weights.pth"
WEIGHTS_PATH = r"best.pth"

# Make RETFound imports available
if RETFOUND_ROOT not in sys.path:
    sys.path.append(RETFOUND_ROOT)

from util.pos_embed import interpolate_pos_embed  # noqa: E402
import models_vit  # noqa: E402

app = FastAPI(title="Hypertension Classification API")

# Allow local dev frontends (Vite/React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    """Create model with RETFound backbone and replace classification head."""
    model = models_vit.RETFound_mae(
        num_classes=num_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )

    # Load upstream checkpoint for positional embedding compatibility
    upstream_ckpt = torch.load(UPSTREAM_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    checkpoint_model = upstream_ckpt["model"]
    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)

    # Replace head and init
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    trunc_normal_(model.head.weight, std=2e-5)

    # Load fine-tuned weights if available
    if os.path.isfile(WEIGHTS_PATH):
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def _build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = _build_transform()
# Assumes training used ImageFolder with classes sorted alphabetically
CLASSES: List[str] = ["Hypertension", "Normal"]
model = _build_model(num_classes=len(CLASSES), device=device)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        image_bytes = file.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, idx = torch.max(probs, dim=0)

    return {
        "prediction": CLASSES[idx.item()],
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)
        },
    }

