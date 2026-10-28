# virtual-fitting-room

Virtual fitting room: custom-trained CNNs and computer vision to fit wearables on a live camera feed.

Users sign in with Google, upload a photo of any wearable (tops, jackets, shoes, hats, glasses, headbands, watches), and see it fitted on their body in real time through the webcam. The item moves with them. Photos and videos can be saved to their account.

## Features

- Live try-on that runs in the browser, so the camera feed never leaves the device
- Upload any wearable and the app detects its category, removes the background and finds its key points
- Smooth tracking with mesh warping, so the item bends with the body
- Arms and hands drawn in front of the item
- Lighting match and soft shadow so the item does not look pasted on
- Guest mode: try items with nothing stored on the server
- Accounts: save photos and videos, view and delete them later

## Where CNNs and computer vision are used

**Trained in this project (CNN):**

| Model | Job | Concept |
|---|---|---|
| Category classifier | Top, shoes, hat, glasses or watch | Image classification, transfer learning |
| Garment cutout | Remove the background | Semantic segmentation (U-Net) |
| Garment keypoints | Find collar, shoulders, hem, sleeve ends | Heatmap regression |

**Pretrained (CNN, used as a tool):**

| Model | Job |
|---|---|
| MediaPipe Pose, Face, Hand | Track the body, face and hands |
| MediaPipe person segmentation | Mark which pixels are the person |

**Written in this project (classic computer vision):**

- Landmark smoothing (One Euro filter)
- Matching garment keypoints to body points
- Mesh warping
- Occlusion compositing (arms in front)
- Lighting and shadow matching
- Real-time rendering

## How it works

```
Upload an item (once)
  classifier -> cutout -> keypoints

Every camera frame (about 30 times a second, in the browser)
  MediaPipe body tracking -> smoothing -> match points -> mesh warp
  -> arms in front -> lighting and shadow -> draw
```

The backend never receives video frames. It only handles login and saved files.

## Tech stack

| Part | Choice |
|---|---|
| ML training | PyTorch on free Kaggle or Colab GPUs |
| Model delivery | ONNX, ONNX Runtime Web |
| Body tracking | MediaPipe Tasks Vision |
| Frontend | React, Vite, TypeScript |
| Styling | CSS Modules and CSS variables |
| Animation | Framer Motion |
| Rendering | PixiJS (WebGL) |
| Backend | FastAPI (Python) |
| Auth | Google Identity Services |
| Database | PostgreSQL on Amazon RDS |
| File storage | Amazon S3 (private bucket, signed URLs) |
| Deployment | Render |

## Project structure

```
virtual-fitting-room/
├── README.md
├── .gitignore
├── ml/                 CNN and computer vision training (Python)
│   ├── requirements.txt
│   └── 00_foundations/
├── backend/            planned: FastAPI, Google login, RDS, S3
└── frontend/           planned: React, MediaPipe, PixiJS
```

New folders are added when each phase starts.

## Roadmap

### Phase 0: CNN foundations
- [x] Convolution from scratch in NumPy (`conv2d`)
- [ ] Padding and stride
- [ ] Multiple channels and multiple filters
- [ ] Activation functions and pooling
- [ ] First CNN in PyTorch on Fashion-MNIST
- [ ] Training loop, loss, accuracy, overfitting

### Phase 1: Category classifier
- [ ] Collect and clean a wearables dataset
- [ ] Fine-tune a pretrained model (transfer learning)
- [ ] Evaluate with a confusion matrix
- [ ] Save the model

### Phase 2: Garment cutout
- [ ] Learn segmentation and the U-Net design
- [ ] Train a small U-Net
- [ ] Evaluate with IoU and Dice score

### Phase 3: Garment keypoints
- [ ] Learn heatmap-based keypoint detection
- [ ] Prepare DeepFashion2 landmarks
- [ ] Train the keypoint network
- [ ] Evaluate keypoint error

### Phase 4: Export models
- [ ] Export all three models to ONNX
- [ ] Shrink the models for browser speed
- [ ] Test inference in the browser

### Phase 5: Frontend and live try-on
- [ ] React and Vite setup, camera access
- [ ] MediaPipe body tracking in the browser
- [ ] Smoothing
- [ ] Place an item on the body
- [ ] Mesh warp using garment keypoints
- [ ] Arms in front of the item
- [ ] Lighting and shadow match
- [ ] Other categories: shoes, hats, glasses, watches

### Phase 6: Backend
- [ ] FastAPI project and health check
- [ ] Google sign-in
- [ ] RDS Postgres: users, garments, media tables
- [ ] S3 bucket and signed upload and download links
- [ ] Per-user limits (file size, video length, item count)
- [ ] Delete item and delete account

### Phase 7: UI polish
- [ ] Palette and design tokens
- [ ] Landing page and animations
- [ ] Catalog, upload flow, saved looks
- [ ] Photo capture and video recording
- [ ] Mobile layout

### Phase 8: Deployment
- [ ] AWS billing alert and IAM least-privilege setup
- [ ] Deploy the backend on Render
- [ ] Deploy the frontend on Render (static site)
- [ ] CORS and environment variables
- [ ] Keep-awake pinger for the backend

### Phase 9: Stretch
- [ ] Size suggestions from body proportions
- [ ] Share links for saved looks
- [ ] Better fit using 3D or generative methods

## Privacy

- The live camera feed is processed on the user's device and is never uploaded
- Guests store nothing on the server
- Saved files are private and shown through short-lived signed links
- Users can delete any item or their whole account

## Limitations

A flat 2D photo cannot match a real 3D fit. The goal is a convincing result: the item follows the body, bends with it and sits correctly behind the arms.

## Setup

ML environment:

```bash
cd ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Backend and frontend setup will be added in their phases.

## Security notes

- Never commit `.env` files or AWS or Google keys
- Keys live only in Render environment variables
