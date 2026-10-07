# Lab 1: System Benchmarks & Deployment Feasibility

## 1. Goal
Benchmark inference latency, memory consumption, and binary storage footprint for baseline classifiers on the Breast Cancer Wisconsin dataset, evaluating suitability across embedded and server hardware tiers.

## 2. Method
Evaluated two classifiers:
- Logistic Regression (`max_iter=1000`, `random_state=42`)
- Random Forest (`n_estimators=100`, `random_state=42`)

Trained on a 70/30 stratified split (`random_state=42`). Runtime latency was captured using median timings over multiple runs, resident memory was measured via `psutil`, and artifacts were serialized using `joblib`.

## 3. Results

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 0.9415 | 0.9357 |
| **Median Training Time (ms)** | 58.32 | 57.26 |
| **Median Inference Latency (ms)** | 0.0192 | 0.8764 |
| **Peak Memory (MB)** | 150.72 | 152.53 |
| **Model Size (Bytes)** | 1,103 | 290,905 |
| **Model Size (KB)** | 1.08 | 284.09 |

### Deployment Evaluation

| Target Tier | Constraints | Logistic Regression | Random Forest |
| :--- | :--- | :--- | :--- |
| **Cloud** | $\ge$ 1 GB RAM, $\le$ 100 ms, $\le$ 500 MB | Compatible | Compatible |
| **Edge** | 256–1024 MB RAM, $\le$ 50 ms, $\le$ 50 MB | Compatible | Compatible |
| **Mobile** | 64–256 MB RAM, $\le$ 20 ms, $\le$ 10 MB | Compatible | Compatible |
| **TinyML** | $\le$ 256 KB RAM, $\le$ 10 ms, $\le$ 100 KB | Compatible (exported C weights) | Incompatible (exceeds size budget) |

## 4. Three Conclusions in Your Own Words
1. **Model footprint diverges despite identical task accuracy:** Both models achieve ~94% test accuracy, yet Random Forest requires over 260x more storage (284 KB vs 1.08 KB) due to storing node structures for 100 individual trees.
2. **Inference latency is the primary operational differentiator:** Offline training times were nearly identical (~57–58 ms), but tree traversal increased prediction latency by more than 45x compared to the linear dot product.
3. **TinyML deployment viability:** Logistic Regression weights can easily fit onto constrained microcontrollers (< 100 KB budget), whereas Random Forest cannot run on TinyML targets without substantial tree pruning or compression.
