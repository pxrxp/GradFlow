# Theory of GradFlow

## 1. The Computation Graph

The engine represents mathematical expressions as a Directed Acyclic Graph (DAG). 

### Node Structure
Every Value object is a node in this graph.
- data: The scalar value computed in the forward pass.
- grad: The derivative dL/d node.
- _parents: Set of parent nodes (dependencies).
- _op: The operation that produced this node.

## 2. Gradient Flow (Backpropagation)

The goal is to calculate how the final output L changes with respect to any input node x.

### The Chain Rule
For any node z created by an operation on x, the gradient is:
dL/dx = dL/dz * dz/dx

Implementation:
def _backward():
    x.grad += (local_derivative) * z.grad
