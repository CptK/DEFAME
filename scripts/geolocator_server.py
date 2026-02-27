"""
Geolocator model server. Loads StreetCLIP once per worker and serves
geolocation requests over HTTP so multiple DEFAME workers can share model
instances without CUDA/fork issues.

Usage:
    python scripts/geolocator_server.py --model geolocal/StreetCLIP --port 5555 --workers 5
"""
import argparse
import base64
import io
import os

import torch
import uvicorn
from PIL import Image as PILImage
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel

app = FastAPI()

# Populated in startup() after fork, so each worker process has its own copy.
processor = None
model = None
device = None

DEFAULT_CHOICES = [
    'Albania', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda',
    'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China',
    'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Estonia',
    'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hungary',
    'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kenya',
    'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
    'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand',
    'Nigeria', 'Norway', 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal',
    'Puerto Rico', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia',
    'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden',
    'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine',
    'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay',
]


class GeolocateRequest(BaseModel):
    image_b64: str
    top_k: int = 10
    choices: list[str] = None


class GeolocateResponse(BaseModel):
    most_likely_location: str
    top_k_locations: list[str]
    text: str


@app.on_event("startup")
async def startup():
    """Load the model inside each worker process after forking.

    Loading here (rather than in __main__) avoids inheriting a CUDA context
    across a fork, which can cause hangs or errors.
    """
    global processor, model, device
    model_name = os.environ.get("GEOLOCATOR_MODEL", "geolocal/StreetCLIP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PID {os.getpid()}] Loading {model_name} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"[PID {os.getpid()}] Geolocator ready.", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/geolocate", response_model=GeolocateResponse)
def geolocate(req: GeolocateRequest):
    choices = req.choices or DEFAULT_CHOICES
    image = PILImage.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")

    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits_per_image.softmax(dim=1)
    confidences = {choices[i]: round(float(prediction[0][i].item()), 2) for i in range(len(choices))}
    top_k = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:req.top_k])
    most_likely = max(top_k, key=top_k.get)

    return GeolocateResponse(
        most_likely_location=most_likely,
        top_k_locations=list(top_k.keys()),
        text=f"The most likely countries where the image was taken are: {top_k}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="geolocal/StreetCLIP")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Pass model name to worker processes via environment variable.
    os.environ["GEOLOCATOR_MODEL"] = args.model

    uvicorn.run(
        "scripts.geolocator_server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
    )
