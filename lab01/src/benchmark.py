import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process(os.getpid())
results = []
accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies.append({"model": name, "accuracy": f"{acc:.4f}"})

    train_times = []
    model.fit(X_train, y_train)
    for _ in range(5):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    median_train_time_ms = np.median(train_times) * 1000

    peak_mem_mb = process.memory_info().rss / (1024 * 1024)

    sample = X_test[0:1]
    infer_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        infer_times.append(time.perf_counter() - t0)
    median_infer_lat_ms = np.median(infer_times) * 1000

    model_path = f"lab01/results/{name}.joblib"
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    results.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "Train Time (ms)": round(median_train_time_ms, 2),
        "Inference Latency (ms)": round(median_infer_lat_ms, 4),
        "Peak Memory (MB)": round(peak_mem_mb, 2),
        "Size (Bytes)": size_bytes,
        "Size (KB)": round(size_kb, 2)
    })

acc_df = pd.DataFrame(accuracies)
acc_df.to_csv("lab01/results/baseline_accuracy.csv", index=False)

res_df = pd.DataFrame(results)
res_df.to_csv("lab01/results/benchmark_metrics.csv", index=False)
print(res_df.to_string(index=False))
