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
  - [1. Model (training & evaluation)](#1-model-training--evaluation)
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

## Project Structure

```
cifar10-cnn-classifier/
├── .venv/                     # Shared virtual environment (gitignored)
├── model/                     # Model training, evaluation & experimentation
│   ├── notebooks/              # Jupyter notebooks (EDA, training, evaluation)
│   ├── src/                    # Reusable Python modules
│   │   ├── data_loader.py       # Dataset loading & preprocessing
│   │   ├── model.py             # CNN architecture definition
│   │   ├── train.py             # CLI training script
│   │   └── evaluate.py          # Metrics, confusion matrix, error analysis
│   ├── saved_models/            # Trained model artifacts (gitignored)
│   └── requirements.txt
│
├── backend/                   # FastAPI inference API
│   ├── main.py
│   ├── routers/
│   │   └── predict.py
│   └── requirements.txt
│
├── frontend/                  # React web demo (Vite)
│   ├── src/
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

### 1. Model (training & evaluation)

```bash
source .venv/Scripts/activate      # if not already active
jupyter notebook model/notebooks/01_cnn_training.ipynb
```

### 2. Backend (API)

```bash
source .venv/Scripts/activate      # if not already active
cd backend
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Frontend (web demo)

```bash
cd frontend
npm install
npm run dev
```

App will be available at `http://localhost:5173`.

## Model Architecture

_To be documented once the architecture is finalized — will include layer diagram, parameter count, and design rationale._

## Results

_To be filled in after training and evaluation:_

- Test accuracy / loss
- Confusion matrix
- Per-class precision & recall
- Training/validation curves

## Known Limitations & Future Improvements

_To be filled in after evaluation — e.g. commonly confused classes, overfitting behavior, and proposed solutions (data augmentation, regularization, architecture changes, transfer learning)._

## Author

Built as part of the Deep Learning / CNN module coursework.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.