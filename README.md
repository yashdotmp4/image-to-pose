

## Requirements

- Python 3.10+
- A virtual environment (recommended)
- CUDA-capable GPU (optional — falls back to CPU)
- Linux


## Setup & Running

**1. Create and activate a virtual environment**

If you have limited disk space on your home partition, create the venv on a scratch partition:

python3 -m venv /scratch/<your-username>/venv
source /scratch/<your-username>/venv/bin/activate


Or in the project directory if you have space:

python3 -m venv venv
source venv/bin/activate


**2. Run the application**

chmod +x run.sh
./run.sh


This will install all dependencies, start the backend API on port 8000, and serve the frontend on port 3000.

**3. Open the application**

```
http://localhost:3000
```

---

## Usage

1. Select an **HRNet model** from the dropdown:
   - **HRNet First** — trained on full COCO 118k dataset (recommended)
   - **HRNet MMA** — trained on COCO + MMA fighter dataset
2. Click **Upload** and select an image of a person
3. Click **Extract Pose**
4. The 3D figure will update to match the detected pose
5. Use the **light direction** and **depth** sliders to adjust the scene
6. Click **Show debug views** to see the 2D keypoint detection and 3D skeleton plots
7. Drag to orbit the 3D viewport, scroll to zoom

---

## Project Structure

```
image-to-pose/
├── api/
│   └── main.py              # FastAPI backend
├── frontend/
│   └── index.html           # Three.js frontend
├── models/
│   ├── hrnet.py             # HRNet architecture
│   └── lifting_network.py   # MartinezNet architecture
├── checkpoints/
│   ├── hrnet-to-test/
│   │   ├── hrnet_first.pth  # HRNet trained on COCO 118k
│   │   └── hrnet_mma.pth    # HRNet trained on COCO + MMA
│   └── martinez-to-test/
│       └── mart_aug_two.pth # Best MartinezNet model
├── requirements.txt
└── run.sh
```

---

## Models

| Model | Description | Performance |
|-------|-------------|-------------|
| hrnet_first | HRNet (width=32) trained on COCO 118k | PCK@0.2: 15.1% |
| hrnet_mma | HRNet (width=32) trained on COCO + MMA dataset | PCK@0.2: 11.6% |
| mart_aug_two | MartinezNet trained on 3DPW HRNet-paired data | MPJPE: ~151mm |

---

## Notes

- The backend API runs on `http://localhost:8000`
- The frontend is served on `http://localhost:3000`
- Press `Ctrl+C` to stop both servers