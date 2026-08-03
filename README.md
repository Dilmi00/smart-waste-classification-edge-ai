# 🗑️ Smart Waste Classification for Edge AI using MLflow & ONNX Runtime

> A lightweight end-to-end computer vision pipeline for waste image classification using **PyTorch**, **MLflow**, **ONNX**, and **ONNX Runtime**, demonstrating MLOps practices and Edge AI optimization.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![ONNX](https://img.shields.io/badge/ONNX-Edge%20Deployment-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Project Overview

Smart Waste Classification is an end-to-end computer vision project that classifies waste images into six recyclable categories using a lightweight MobileNetV3-Small model.

The project demonstrates modern machine learning engineering practices by incorporating:

- Transfer Learning
- MLflow Experiment Tracking
- Model Versioning
- ONNX Model Export
- ONNX Runtime Inference
- Edge AI Performance Benchmarking

The optimized ONNX model achieved over **4× faster inference** while maintaining identical predictions compared to the original PyTorch model.

---

# 🚀 Features

- ✅ Waste image classification
- ✅ MobileNetV3-Small Transfer Learning
- ✅ Class imbalance handling
- ✅ MLflow experiment tracking
- ✅ Hyperparameter comparison
- ✅ Best model checkpointing
- ✅ ONNX model export
- ✅ ONNX Runtime inference
- ✅ Edge AI latency benchmarking

---

# 🏗️ Project Architecture

```text
                   Waste Image Dataset
                           │
                           ▼
                Data Preprocessing
                           │
                           ▼
         MobileNetV3-Small Transfer Learning
                           │
                           ▼
             MLflow Experiment Tracking
                           │
                           ▼
                 Best Model Selection
                           │
                           ▼
                   Export to ONNX
                           │
                           ▼
              ONNX Runtime Inference
                           │
                           ▼
             Performance Benchmarking
```

---

# 📂 Dataset

Dataset:

**TrashNet Dataset**

Classes:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

Images are automatically split into:

- Training (70%)
- Validation (15%)
- Test (15%)

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Model | MobileNetV3-Small |
| Experiment Tracking | MLflow |
| Model Optimization | ONNX |
| Inference Engine | ONNX Runtime |
| Metrics | Accuracy, Macro F1 |
| Version Control | Git & GitHub |

---

# 📁 Project Structure

```text
smart-waste-classification-edge-ai/
│
├── artifacts/
├── models/
├── notebooks/
├── screenshots/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── export_onnx.py
│   ├── infer_onnx.py
│   ├── benchmark.py
│   ├── model.py
│   └── ...
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/smart-waste-classification-edge-ai.git

cd smart-waste-classification-edge-ai
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Training

```bash
python -m src.train --epochs 5 --batch-size 32 --lr 0.001
```

Example experiments

| Learning Rate | Batch Size |
|---------------|-----------|
| 0.001 | 32 |
| 0.0005 | 32 |

Experiments are automatically tracked using MLflow.

---

# 📈 MLflow Experiment Tracking

Launch MLflow UI

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open

```
http://127.0.0.1:5000
```

MLflow logs:

- Hyperparameters
- Training Loss
- Validation Loss
- Validation Accuracy
- Validation Macro F1
- Best Model Checkpoint

---

# 🔄 Export to ONNX

```bash
python -m src.export_onnx
```

The trained PyTorch model is converted into an ONNX model for optimized edge deployment.

---

# ⚡ ONNX Runtime Inference

```bash
python -m src.infer_onnx --image path/to/image.jpg
```

---

# 🚀 Performance Benchmark

```bash
python -m src.benchmark
```

### Benchmark Results

| Metric | PyTorch | ONNX Runtime |
|---------|---------:|-------------:|
| Inference Latency | 7.95 ms | 1.90 ms |
| Model Size | 5.94 MB | 0.31 MB |
| Prediction Agreement | ✓ | ✓ |
| Speedup | - | **4.18× Faster** |

---

# 📈 Model Performance

| Metric | Value |
|---------|------:|
| Best Validation Accuracy | **84.70%** |
| Best Validation Macro F1 | **0.8177** |
| Test Accuracy | **85.00%** |
| Test Macro F1 | **0.8184** |

---

# 📷 Results

## MLflow Experiment Tracking

Add screenshots:

```
screenshots/mlflow_dashboard.png
screenshots/val_accuracy.png
screenshots/val_loss.png
screenshots/val_f1_macro.png
```

## Benchmark Results

Add

```
screenshots/benchmark_results.png
```

---

# 💡 Edge AI Optimization

The trained MobileNetV3 model was exported to ONNX format and executed using ONNX Runtime.

Compared to the original PyTorch model, ONNX Runtime achieved:

- 4.18× faster inference
- ~95% smaller serialized model size
- Identical prediction outputs

These improvements make the model more suitable for deployment on resource-constrained edge devices.

---

# 🖥️ Web Application

The project includes an interactive Streamlit application that allows users to upload a waste image and receive a real-time prediction using the optimized ONNX model.

Run the application:

```bash
streamlit run app.py
```

Features:

- Upload waste images
- Real-time classification
- Confidence score
- Class probability distribution
- ONNX Runtime inference

---

# 🔮 Future Improvements

- Deploy with FastAPI
- Docker containerization
- TensorRT optimization
- Raspberry Pi deployment
- Real-time webcam inference
- CI/CD pipeline using GitHub Actions

---



# 📄 License

This project is licensed under the MIT License.