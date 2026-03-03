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

The project includes a clinical classification demonstration in [demo.ipynb](demo.ipynb). It utilizes a real-world medical dataset ([medical_data.csv](medical_data.csv)) to train a model for malignant vs. benign diagnosis.

### Training Workflow
- **Data Loading**: Custom CSV utility for feature extraction and label formatting.
- **Model**: An MLP architecture (3 inputs, two hidden layers of 6, 1 output).
- **Optimization**: The model is trained using **Batch Gradient Descent** over 500 epochs.
- **Evaluation**: Final model weights are used to run inference on clinical samples, outputting raw predictions and mapped diagnoses.

## Technical Notes
- **Loss Function**: Mean Squared Error (MSE).
- **Initialization**: Gaussian scaling is used for weight initialization to prevent dead neurons.
- **Gradient Reset**: Explicit `model.zero_grad()` calls at the start of each epoch to prevent illegal gradient accumulation.

---
*Developed as a learning project to explore the inner workings of automatic differentiation and neural network optimization.*
