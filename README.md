# 🧠 GradFlow

GradFlow is a from-scratch implementation of a deep learning stack, progressing from a base scalar-valued autograd engine to a Multi-Layer Perceptron (MLP) architecture. The project demonstrates the transition from fundamental calculus and automatic differentiation to practical model training on clinical data.

## Implementation Overview

The repository is structured to show the building blocks of a neural network framework:

### 1. Scalar Engine ([engine.py](gradflow/engine.py))
The foundation of the project is the `Value` class, which handles the core mechanics of automatic differentiation. 
- **Computational Graph**: Every mathematical operation builds a Directed Acyclic Graph (DAG) of `Value` nodes.
- **Backpropagation**: Gradients are computed using the multivariate chain rule, implemented through recursive `_backward` closures.
- **Topological Order**: A custom sorting algorithm ensures that gradients flow correctly from the output back to the leaf nodes.

### 2. Tensor Abstraction ([tensor.py](gradflow/tensor.py))
To handle multi-dimensional data, a recursive `Tensor` wrapper was implemented to bridge the gap between matrix math and scalar-level tracking.
- **Recursive Mapping**: Operations are dispatched through the tensor grid, mapping matrix-level functions down to individual `Value` nodes.
- **Matrix Operations**: Includes support for matrix multiplication (`@`), transposition (`T()`), and element-wise arithmetic.
- **Auto-differentiation Boundary**: The `.sum()` reduction collapses high-dimensional tensors into a root `Value` node for starting the backpropagation pass.

### 3. Neural Network Modules ([nn/](gradflow/nn/))
A modular hierarchy for constructing deep architectures:
- **Neuron**: Individual Perceptron logic implementing $activation(\sum w_i x_i + b)$.
- **Layer**: Manages parallel neuron execution.
- **MLP**: Organizes layers into a sequential architecture with parameter collection across the hierarchy.

## Practical Application: Breast Cancer Diagnosis

The project includes a clinical classification demonstration in [demo.ipynb](demo.ipynb). It utilizes the [Wisconsin Breast Cancer Diagnostic (WDBC)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset ([medical_data.csv](medical_data.csv)) to train a model for malignant vs. benign diagnosis.

### Dataset Overview
The dataset contains 569 clinical samples, each with features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of cell nuclei. The goal is to predict the diagnosis (M = Malignant, B = Benign).

<img width="1300" height="697" alt="image" src="https://github.com/user-attachments/assets/f2fd6dbb-1fd3-4810-886d-c83636a29824" />

In our demonstration, we utilize three key features:
- **Radius Mean**: Mean of distances from center to points on the perimeter.
- **Texture Mean**: Standard deviation of gray-scale values.
- **Smoothness Mean**: Local variation in radius lengths.

### Training Workflow
- **Data Loading**: Custom CSV utility for feature extraction and label formatting.
- **Model**: An MLP architecture (3 inputs, two hidden layers of 6 nodes, 1 output).
- **Optimization**: The model is trained using **Batch Gradient Descent** over 500 epochs with a train-test split for unbiased evaluation.
- **Evaluation**: Final model weights are used to run inference on clinical samples, producing Raw Model Scores mapped to diagnosis labels.

### Model Persistence
Pre-calculated model weights and z-score normalization parameters (means/stds) are provided in [output_model_weights.json](output_model_weights.json). This allows the system to perform immediate inference without retraining.

## 🚀 CLI tool for Predictions

The project includes a standalone CLI tool, [predict.py](predict.py), for running inference on saved model weights without a notebook environment.

### Usage
```bash
$ python predict.py --radius 15.22 --texture 30.62 --smoothness 0.1048
Model loaded from output_model_weights.json

--- GradFlow Diagnosis Report ---
Input Features: Radius=15.22, Texture=30.62, Smoothness=0.1048
Raw Model Score: 0.5831
Diagnosis: MALIGNANT
---------------------------------
```
- **Inputs**: Requires mean radius, texture, and smoothness values.
- **Persistence**: Automatically loads pre-trained weights and normalization parameters from [output_model_weights.json](output_model_weights.json).

## 📚 Further Reading

For a deeper dive into the mathematical foundations of this implementation, refer to:
- **[THEORY.md](THEORY.md)**: A modular guide covering the autograd engine, tensor abstractions, and neural network optimization principles.

## Technical Notes
- **Loss Function**: Mean Squared Error (MSE).
- **Initialization**: Gaussian scaling is used for weight initialization to prevent dead neurons.
- **Gradient Reset**: Explicit `model.zero_grad()` calls at the start of each epoch to prevent illegal gradient accumulation.

---
*Developed as a learning project to explore the inner workings of automatic differentiation and neural network optimization.*
