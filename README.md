# MLE_practice
Structured MLE Curriculum with a modular class-based codebase.

Phase 1: Foundations of Machine Learning (4 Weeks)
1. Time Series Forecasting (ARIMA, SARIMA, Prophet)
📌 Concepts:

Stationarity & differencing

AR, MA, ARMA, ARIMA, SARIMA

Prophet Model

📌 Code Implementation:

Build ARIMA, SARIMA models from scratch (without libraries like statsmodels)

Implement Prophet-like forecasting logic

2. Traditional ML Models (No Libraries, Pure Python)
📌 Concepts:

Implement Linear Regression, Logistic Regression (Gradient Descent)

Implement Decision Tree (Gini, Entropy)

Implement Random Forest (Bootstrap Aggregation)

Implement XGBoost (Gradient Boosting)

📌 Code Implementation:

Class-based implementation for each ML model

Write a unified ML Trainer module

Phase 2: Deep Learning Fundamentals (4 Weeks)
3. Neural Networks from Scratch
📌 Concepts:

Forward & Backpropagation

Loss Functions & Optimizers

Multi-Layer Perceptron (MLP)

📌 Code Implementation:

Implement Fully Connected NN (MLP) from scratch (NumPy only)

Modular Neural Network class with backpropagation

4. PyTorch & TensorFlow for Model Training
📌 Concepts:

Tensors, Autograd, Dataset Handling

Training loops, Model Checkpointing

Save & Load Models for Inference

📌 Code Implementation:

Class-based PyTorch/TensorFlow model trainer

Save models (.pth/.h5) and use for inference

Phase 3: Advanced AI Topics (6 Weeks)
5. Computer Vision (Custom CNN Model)
📌 Concepts:

Custom CNN Architectures

Transfer Learning & Fine-tuning

Image Classification, Object Detection

📌 Code Implementation:

Build CNN from scratch in PyTorch/TensorFlow

Implement a modular Vision Model Trainer class

6. NLP (Transformer, BERT, Semantic Search, Keyword Extraction)
📌 Concepts:

Transformer Model & Attention

BERT from scratch

Word Embeddings & Semantic Search

Topic Modeling

📌 Code Implementation:

Build Transformer & BERT from scratch

Implement Semantic Search using embeddings

Class-based Keyword Extraction & Topic Ranking module

Phase 4: Deployment & MLOps (3 Weeks)
7. Deployment using Docker
📌 Concepts:

Containerizing models

Creating API with Flask/FastAPI

Serving models with TensorFlow Serving/TorchServe

📌 Code Implementation:

Dockerized ML/DL API with FastAPI

Model Serving with Docker + Flask/FastAPI
