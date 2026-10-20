from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import cv2

DATA = Path("data/colored_images")
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
Path("figures").mkdir(exist_ok=True)

print(pd.read_csv("data/train.csv").head())

# collect image paths per class
files = {c: sorted(f for f in (DATA / c).iterdir() if f.is_file()) for c in CLASSES}
counts = {c: len(v) for c, v in files.items()}
print(counts)

# bar chart of class counts
plt.figure(figsize=(7, 4))
plt.bar(counts.keys(), counts.values())
plt.title("Images per class")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/class_counts.png", dpi=150)

# 2 sample images per class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for col, c in enumerate(CLASSES):
    for row in range(2):
        img = cv2.cvtColor(cv2.imread(str(files[c][row])), cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(c)
plt.tight_layout()
plt.savefig("figures/samples.png", dpi=150)