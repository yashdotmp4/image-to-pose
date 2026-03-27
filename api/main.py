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
sys.path.append('C:/Users/yasha/FYPP/image-to-pose')
from models.hrnet import HRNet
from models.lifting_network import MartinezNet

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hrnet = HRNet(num_keypoints=17).to(device)
hrnet.load_state_dict(torch.load(
    'C:/Users/yasha/FYPP/image-to-pose/checkpoints/hrnet_best.pth',
    map_location=device, weights_only=False))
hrnet.eval()

martinez = MartinezNet(num_joints_in=17, num_joints_out=17).to(device)
martinez.load_state_dict(torch.load(
    'C:/Users/yasha/FYPP/image-to-pose/checkpoints/best_model.pth',
    map_location=device, weights_only=False))
martinez.eval()

transform = T.Compose([
    T.Resize((256, 192)),
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
            x = x * orig_w / 48
            y = y * orig_h / 64
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

        # lift to 3D
        inp_2d = torch.tensor(keypoints_2d_norm.reshape(1, -1), dtype=torch.float32).to(device)
        pose_3d = martinez(inp_2d)[0].cpu().numpy()

    pose_3d = pose_3d * 2
    pose_3d[:, 1] += 1.0

    light_direction = [1, 2, 1]

    return JSONResponse({
        "keypoints_3d": pose_3d.tolist(),
        "light_direction": light_direction,
        "debug_image": debug_b64
    })