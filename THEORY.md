# Theory of GradFlow

This documentation covers the mathematical and architectural foundations of the GradFlow engine, following its progression from scalar autograd to full neural network training.

## Table of Contents

### 1. [The Autograd Engine](theory/01_autograd.md)
The core `Value` class, computational graphs, multivariate chain rule, and topological sorting.

### 2. [Tensors Abstraction](theory/02_tensors.md)
Recursive multidimensional wrappers, matrix operations (MatMul, Transpose), and reduction boundaries.

### 3. [Neural Network Components](theory/03_nn_components.md)
Implementation of Neurons, Layers, and Multi-Layer Perceptrons (MLP).

### 4. [Learning Principles](theory/04_learning.md)
Loss functions (MSE), Gradient Descent optimization, Learning Rates, and Normalization.

### 5. [Practical Training Workflow](theory/05_workflow.md)
Step-by-step breakdown of the training loop used in `demo.ipynb`.

---
*For a quick start with the code, see the [README.md](../README.md).*
