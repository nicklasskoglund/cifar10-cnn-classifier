# CIFAR-10 CNN Image Classifier

> A convolutional neural network built from scratch to classify images from the CIFAR-10 dataset, served through a FastAPI backend and demonstrated via a React web application.

**Status:** 🚧 In development — school project (Deep Learning / CNN module)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [0. Environment setup (once)](#0-environment-setup-once)
  - [0.5 Dataset setup (once)](#05-dataset-setup-once)
  - [1. Model (training & evaluation)](#1-model-training--evaluation)
  - [1.5 Model (CLI alternative)](#15-model-cli-alternative)
  - [2. Backend (API)](#2-backend-api)
  - [3. Frontend (web demo)](#3-frontend-web-demo)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Known Limitations & Future Improvements](#known-limitations--future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview

This project implements an end-to-end deep learning workflow:

1. **Train** a Convolutional Neural Network (CNN) on the CIFAR-10 image dataset.
2. **Evaluate** the model's performance and analyze its weaknesses (e.g. commonly confused classes).
3. **Iterate** on the architecture/hyperparameters to improve accuracy.
4. **Serve** the trained model through a REST API (FastAPI).
5. **Demonstrate** the model interactively through a React web application, where a user can upload an image and see the model's prediction along with its confidence across all classes.

This was built as a school project to demonstrate practical understanding of CNNs — from data preprocessing to model deployment — rather than as a production-grade system.

## Dataset

**[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)** consists of 60,000 32x32 color images across 10 mutually exclusive classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

- 50,000 training images
- 10,000 test images
- 6,000 images per class (balanced dataset)

> _A more detailed dataset analysis (class distribution, sample images, pixel statistics) is documented in [`model/notebooks/01_cnn_training.ipynb`](model/notebooks/01_cnn_training.ipynb)._

**Note on data loading:** this project does *not* use `tf.keras.datasets.cifar10.load_data()`. In practice, that function's built-in download/cache handling proved unreliable (slow/unstable connections to the original dataset host, and inconsistent local-cache validation even with a verified checksum match). Instead, the dataset is downloaded once manually and read directly from the extracted pickle files via a small custom loader in [`model/src/data_loader.py`](model/src/data_loader.py). See [0.5 Dataset setup](#05-dataset-setup-once) below for the one-time download step.

## Project Structure

```
cifar10-cnn-classifier/
├── .venv/                     # Shared virtual environment (gitignored)
├── model/                     # Model training, evaluation & experimentation
│   ├── notebooks/              # Jupyter notebooks (EDA, training, evaluation)
│   ├── src/                    # Reusable Python modules
│   │   ├── __init__.py          # Makes model/src a Python package
│   │   ├── data_loader.py       # Dataset loading & preprocessing
│   │   ├── model.py             # CNN architecture definition
│   │   ├── train.py             # Reusable training function (used by notebook & CLI)
│   │   └── evaluate.py          # Metrics, confusion matrix, error analysis
│   ├── saved_models/            # Trained model artifacts (gitignored)
│   └── requirements.txt
│
├── run_pipeline.py             # CLI entry point: runs the full train+evaluate pipeline
│                                # without opening Jupyter (uses model/src modules)
│
├── backend/                   # FastAPI inference API
│   ├── main.py
│   ├── routers/
│   │   └── predict.py
│   └── requirements.txt
│
├── frontend/                  # React web demo (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx    # File selection, image preview, Predict button
│   │   │   └── PredictionResult.jsx # Predicted class + per-class confidence bar chart
│   │   ├── App.jsx                   # Main component: upload/prediction/loading/error state
│   │   └── App.css                   # Dark-mode styling
│   └── package.json
│
└── docs/                      # Presentation materials & supporting images
```

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Python, TensorFlow / Keras |
| Experimentation | Jupyter Notebook |
| Backend API | FastAPI, Uvicorn |
| Frontend | React (Vite) |
| Version control | Git / GitHub (feature branch workflow) |

## Getting Started

> **Note on hardware:** Training runs on CPU. TensorFlow dropped native GPU support on Windows starting with version 2.11 (GPU training now requires WSL2). To keep the workflow simple and fully within native Windows + GitBash, this project trains on CPU — CIFAR-10 is small enough that this remains practical (~1-3 min/epoch).

### 0. Environment setup (once)

A single virtual environment at the project root is shared between `model/` and `backend/`:

```bash
python -m venv .venv
source .venv/Scripts/activate      # GitBash on Windows
pip install -r model/requirements.txt
pip install -r backend/requirements.txt
```

### 0.5 Dataset setup (once)

Download the dataset from Kaggle: **[cifar10-python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python)** (this mirrors the original CIFAR-10 pickle format).

Extract it so the folder structure looks like this:

```
~/.keras/datasets/cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

> This path is used regardless of operating system — on Windows via GitBash, `~` resolves to `C:\Users\<you>\`. No other setup is needed; `data_loader.py` reads these files directly.

### 1. Model (training & evaluation)

```bash
source .venv/Scripts/activate      # if not already active
jupyter notebook model/notebooks/01_cnn_training.ipynb
```

### 1.5 Model (CLI alternative)

As an alternative to the notebook, the full training + evaluation pipeline can be run directly from the terminal — useful for quickly re-running everything after a code change, without opening Jupyter:

```bash
source .venv/Scripts/activate      # if not already active
python run_pipeline.py
```

This loads the dataset, builds and trains the model (with `EarlyStopping` and `ModelCheckpoint`), plots the training/validation curves, and evaluates the model against the test set — printing the final test accuracy/loss and displaying a confusion matrix.

> Note: results vary slightly between runs due to random weight initialization, data augmentation, and batch shuffling — no fixed random seed is used, so each run is an independent experiment.

### 2. Backend (API)

```bash
source .venv/Scripts/activate      # if not already active
cd backend
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Frontend (web demo)

> **Note:** the backend (step 2) must be running at the same time — the frontend has no logic of its own and simply calls the backend's `/predict` endpoint.

```bash
cd frontend
npm install
npm run dev
```

App will be available at `http://localhost:5173`.

The demo lets you upload an image, then displays the model's predicted class along with a confidence bar chart across all 10 CIFAR-10 classes. It's a minimal interface built for the live presentation demo — not a production UI.

## Model Architecture

The model is a CNN built from scratch (no transfer learning), consisting of 3 convolutional blocks with increasing filter depth, followed by a dense classification head:

- **Conv Block 1:** 2× Conv2D (32 filters, 3×3) → BatchNormalization → MaxPooling2D → Dropout (0.25)
- **Conv Block 2:** 2× Conv2D (64 filters, 3×3) → BatchNormalization → MaxPooling2D → Dropout (0.25)
- **Conv Block 3:** 2× Conv2D (128 filters, 3×3) → BatchNormalization → MaxPooling2D → Dropout (0.25)
- **Classification head:** Flatten → Dense (128, ReLU) → Dropout (0.5) → Dense (10, Softmax)

**Total parameters:** 552,362

**Data augmentation:** applied as the first layer(s) of the model (active only during training, not inference) — `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.1)`. These add 0 trainable parameters, since they're deterministic image transformations rather than learned layers.

**Design rationale:**
- BatchNormalization after each Conv2D speeds up and stabilizes training, which matters especially on CPU.
- Dropout at increasing rates (0.25 in conv blocks, 0.5 in the dense head) helps prevent overfitting on the relatively small 32×32 images.
- Filter depth increases (32 → 64 → 128) while spatial dimensions shrink (via MaxPooling), following the standard CNN pattern of trading spatial resolution for feature richness.

**Training configuration:** Adam optimizer, categorical cross-entropy loss, up to 30 epochs with `EarlyStopping` (`monitor="val_loss"`, `patience=5`, `restore_best_weights=True`) and `ModelCheckpoint` (saves best weights to `model/saved_models/best_model.keras`).

## Results

- **Test accuracy:** 83.13%
- **Test loss:** 0.5024

Training and validation curves, along with the full confusion matrix and per-class breakdown, are documented in [`model/notebooks/01_cnn_training.ipynb`](model/notebooks/01_cnn_training.ipynb).

> Note: since no fixed random seed is used, re-running training (via the notebook or `run_pipeline.py`) will produce slightly different results each time — typically within a few percentage points of the figures above.

## Known Limitations & Future Improvements

- **CPU-only training:** training runs on CPU (~1-3 min/epoch), since TensorFlow dropped native GPU support on Windows from version 2.11 onward without WSL2. This keeps epoch counts and architecture size practical but limits how large the model or dataset could realistically grow within the project's constraints.
- **Result variance:** no fixed random seed is used, so accuracy/loss vary slightly between training runs (weight initialization, data augmentation, and batch order are all stochastic).
- **Hardcoded API URL:** the frontend calls `http://localhost:8000/predict` directly, with no environment variable configuration — sufficient for local development and the live demo, but would need to be made configurable for any real deployment.
- **Potential future improvements:** transfer learning (e.g. a pretrained backbone), more aggressive data augmentation, hyperparameter tuning (learning rate schedules, filter counts), or a deeper architecture — none of which were pursued here given the scope of a course project.

## Author

Built as part of the Deep Learning / CNN module coursework.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.