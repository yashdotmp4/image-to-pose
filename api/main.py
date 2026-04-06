from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import io
import sys
import base64
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hrnet import HRNet
from models.lifting_network import MartinezNet

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')
print(f'Base dir: {BASE_DIR}')

hrnet = HRNet(num_keypoints=17).to(device)
hrnet.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'checkpoints/hrnet_best.pth'),
    map_location=device, weights_only=False))
hrnet.eval()

martinez = MartinezNet(num_joints_in=28, num_joints_out=28).to(device)
martinez.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'checkpoints/best_model.pth'),
    map_location=device, weights_only=False))
martinez.eval()

transform = T.Compose([
    T.Resize((384, 288)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_w, orig_h = img.size

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps = hrnet(inp)[0].cpu().numpy()
        keypoints_2d = []
        for i in range(17):
            hm = heatmaps[i]
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            y, x = idx
            x = x * orig_w / 72
            y = y * orig_h / 96
            keypoints_2d.append([x, y])

        keypoints_2d = np.array(keypoints_2d)

        # draw 2D debug image
        SKELETON_2D = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        debug_img = Image.open(io.BytesIO(contents)).convert("RGB")
        draw = ImageDraw.Draw(debug_img)
        for x, y in keypoints_2d:
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red')
        for a, b in SKELETON_2D:
            draw.line([(keypoints_2d[a][0], keypoints_2d[a][1]),
                       (keypoints_2d[b][0], keypoints_2d[b][1])], fill='lime', width=2)
        buf = io.BytesIO()
        debug_img.save(buf, format='JPEG')
        debug_b64 = base64.b64encode(buf.getvalue()).decode()

        # normalise relative to left hip
        root = keypoints_2d[11:12, :]
        keypoints_2d_norm = keypoints_2d - root

        torso_size = np.linalg.norm(keypoints_2d[5] - keypoints_2d[11]) + np.linalg.norm(keypoints_2d[6] - keypoints_2d[12])
        torso_size = max(torso_size / 2, 1e-6)
        keypoints_2d_norm = keypoints_2d_norm / torso_size

        # pad 17 joints to 28
        padded = np.zeros((28, 2))
        padded[:17] = keypoints_2d_norm
        inp_2d = torch.tensor(padded.reshape(1, -1), dtype=torch.float32).to(device)

        # lift to 3D
        pose_3d = martinez(inp_2d)[0].cpu().numpy()  # (28, 3)
        pose_3d = pose_3d[:17]  # take first 17

    # 3D debug
    SKELETON_3D = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Front View')
    ax1.scatter(pose_3d[:, 0], pose_3d[:, 1], c='red', s=30)
    for a, b in SKELETON_3D:
        ax1.plot([pose_3d[a,0], pose_3d[b,0]], [pose_3d[a,1], pose_3d[b,1]], 'lime')
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax2.set_title('Side View')
    ax2.scatter(pose_3d[:, 2], pose_3d[:, 1], c='red', s=30)
    for a, b in SKELETON_3D:
        ax2.plot([pose_3d[a,2], pose_3d[b,2]], [pose_3d[a,1], pose_3d[b,1]], 'lime')
    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    plt.tight_layout()
    buf3d = io.BytesIO()
    plt.savefig(buf3d, format='JPEG', bbox_inches='tight')
    plt.close()
    debug_3d_b64 = base64.b64encode(buf3d.getvalue()).decode()

    pose_3d[:, 0] = -pose_3d[:, 0]
    pose_3d[:, 1] = -pose_3d[:, 1] + 1.0
    pose_3d = pose_3d * 0.8
    pose_3d[:, 1] += 0.8

    light_direction = [1, 2, 1]

    return JSONResponse({
        "keypoints_3d": pose_3d.tolist(),
        "light_direction": light_direction,
        "debug_image": debug_b64,
        "debug_3d": debug_3d_b64
    })