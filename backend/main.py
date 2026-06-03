# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import base64
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI(title="Brain Tumor Detection API with XAI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

device = torch.device("cpu")
CLASSES = ['GLIOMA', 'MENINGIOMA', 'NOTUMOR', 'PITUITARY']

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256), nn.ReLU(),
    nn.Dropout(0.4), nn.Linear(256, 4)
)

try:
    model.load_state_dict(torch.load('resnet_best.pth', map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading weights: {e}")

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Reusable Core Logic for both Single and Batch
async def process_image_core(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    grayscale_cam = cam(input_tensor=tensor, targets=None)[0, :]
    img_resized = np.array(image.resize((224, 224)))
    rgb_img = np.float32(img_resized) / 255
    heatmap_visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    img_pil = Image.fromarray(heatmap_visualization)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {
        "filename": file.filename,
        "diagnosis": CLASSES[predicted_idx.item()],
        "confidence": round(confidence.item() * 100, 2),
        "heatmap": heatmap_base64
    }

# Endpoint 1: Single Image
@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        return await process_image_core(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Multi-Image Batch
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if file.content_type.startswith('image/'):
            try:
                res = await process_image_core(file)
                results.append(res)
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})
    return {"batch_results": results}

@app.get("/")
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return {"error": "index.html not found! Ensure your folder structure is correct."}
    return FileResponse(frontend_path)

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    from threading import Timer
    def open_browser(): webbrowser.open_new("http://127.0.0.1:8000")
    Timer(1.5, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)