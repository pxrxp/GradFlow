from typing import List
from gradflow.tensor import Tensor

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> List[Tensor]:
        return []
