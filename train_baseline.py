from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
Path("figures").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

X = np.load("features/X.npy")
y = np.load("features/y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, class_weight="balanced"),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=CLASSES, digits=3))

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=30)
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)

joblib.dump(model, "models/svm.joblib")
print("Saved model and confusion matrix")