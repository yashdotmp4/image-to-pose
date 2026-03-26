import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import sys
sys.path.append('/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose')
from models.hrnet import HRNet
from models.lifting_network import MartinezNet

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def infer(img_path, hrnet_checkpoint, martinez_checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load HRNet
    hrnet = HRNet(num_keypoints=17).to(device)
    hrnet.load_state_dict(torch.load(hrnet_checkpoint, map_location=device))
    hrnet.eval()

    # load MartinezNet
    martinez = MartinezNet(num_joints_in=17, num_joints_out=17).to(device)
    martinez.load_state_dict(torch.load(martinez_checkpoint, map_location=device))
    martinez.eval()

    # load image
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    transform = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # stage 1 - 2D keypoints
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

        # normalise 2D keypoints relative to left hip (joint 11)
        root = keypoints_2d[11:12, :]
        keypoints_2d_norm = keypoints_2d - root

        # stage 2 - lift to 3D
        inp_2d = torch.tensor(keypoints_2d_norm.reshape(1, -1), dtype=torch.float32).to(device)
        pose_3d = martinez(inp_2d)[0].cpu().numpy()  # (17, 3)

    # draw 2D result
    draw = ImageDraw.Draw(img)
    for x, y in keypoints_2d:
        draw.ellipse((x-4, y-4, x+4, y+4), fill='red')
    for a, b in SKELETON:
        draw.line([(keypoints_2d[a][0], keypoints_2d[a][1]),
                   (keypoints_2d[b][0], keypoints_2d[b][1])], fill='lime', width=2)

    out_2d = img_path.replace('.jpg', '_2d_output.jpg')
    img.save(out_2d)
    print(f'2D result saved to {out_2d}')
    print(f'3D keypoints (17x3):\n{pose_3d}')

    return keypoints_2d, pose_3d

if __name__ == '__main__':
    kps_2d, kps_3d = infer(
        img_path='/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose/checkpoints/test_img.jpg',
        hrnet_checkpoint='/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose/checkpoints/hrnet_best.pth',
        martinez_checkpoint='/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose/checkpoints/best_model.pth'
    )