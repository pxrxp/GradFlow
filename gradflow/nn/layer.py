from typing import List, Union
from gradflow.tensor import Tensor
from gradflow.nn.base import Module
from gradflow.nn.neuron import Neuron

class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x: Union[List, Tensor]) -> Tensor:
        out_vals = [n(x).data for n in self.neurons]
        return Tensor(out_vals)

    def parameters(self) -> List[Tensor]:
        return [parameter for n in self.neurons for parameter in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"
