import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = np.load("features/X.npy")
y = np.load("features/y.npy")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # same split as train.py
)

candidates = {
    "SVM": (
        make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.0001]},
    ),
    "LogisticRegression": (
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        {"logisticregression__C": [0.001, 0.01, 0.1]},
    ),
}

for name, (pipe, grid) in candidates.items():
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    print(f"{name}: best params {search.best_params_}")
    print(f"  accuracy {accuracy_score(y_test, pred):.4f} | macro F1 {f1_score(y_test, pred, average='macro'):.4f}")