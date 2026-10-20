from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision import models, transforms

DATA = Path("data/colored_images")
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
Path("features").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# pretrained ResNet50 with its final layer removed -> 2048 numbers per image
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device).eval()

normalize = transforms.Compose([
    transforms.ToTensor(),  # pixels 0-255 -> 0-1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet mean/std
])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)       # contrast boost on lightness only
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return normalize(img)

paths, labels = [], []
for label, c in enumerate(CLASSES):
    for f in sorted((DATA / c).iterdir()):
        if f.is_file():
            paths.append(f)
            labels.append(label)

features = []
batch = 32
with torch.no_grad():
    for i in range(0, len(paths), batch):
        x = torch.stack([preprocess(p) for p in paths[i:i + batch]]).to(device)
        features.append(model(x).cpu().numpy())
        if (i // batch) % 10 == 0:
            print(f"{i}/{len(paths)}")

X = np.concatenate(features)
y = np.array(labels)
np.save("features/X.npy", X)
np.save("features/y.npy", y)
print("Saved", X.shape, y.shape)