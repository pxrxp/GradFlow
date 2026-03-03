import random
import math
from typing import List, Union
from gradflow.tensor import Tensor
from gradflow.nn.base import Module

class Neuron(Module):
    def __init__(self, n_inputs: int, use_nonlinear: bool = True):
        scale = math.sqrt(2.0/n_inputs) if use_nonlinear else math.sqrt(1.0/n_inputs)
        self.w = Tensor([random.gauss(0, scale) for _ in range(n_inputs)])
        self.b = Tensor(0.0)
        self.use_nonlinear = use_nonlinear

    def __call__(self, x: Union[List, Tensor]) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x)
        activation = (self.w * x).sum() + self.b
        return activation.relu() if self.use_nonlinear else activation

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.use_nonlinear else 'Linear'}Neuron({len(self.w.data)})"
