from typing import List, Union
from gradflow.tensor import Tensor
from gradflow.nn.base import Module
from gradflow.nn.layer import Layer

class MLP(Module):
    def __init__(self, n_inputs: int, layer_outputs: List[int]):
        layer_sizes = [n_inputs] + layer_outputs
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], use_nonlinear=(i!=len(layer_outputs)-1)) for i in range(len(layer_outputs))]

    def __call__(self, x: Union[List, Tensor]) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"
