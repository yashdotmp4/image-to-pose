from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io

app = FastAPI()

# allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # read image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # placeholder response for now
    keypoints_3d = [[0, 1.7, 0], [-0.1, 1.75, 0], [0.1, 1.75, 0],
                    [-0.2, 1.7, 0], [0.2, 1.7, 0], [-0.3, 1.4, 0],
                    [0.3, 1.4, 0], [-0.5, 1.0, 0], [0.5, 1.0, 0],
                    [-0.6, 0.6, 0], [0.6, 0.6, 0], [-0.2, 0.9, 0],
                    [0.2, 0.9, 0], [-0.2, 0.5, 0], [0.2, 0.5, 0],
                    [-0.2, 0.0, 0], [0.2, 0.0, 0]]
    
    light_direction = [1, 2, 1]
    
    return JSONResponse({
        "keypoints_3d": keypoints_3d,
        "light_direction": light_direction
    })