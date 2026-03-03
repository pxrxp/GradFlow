import math
from typing import Set, Tuple, Union, Callable, List

class Value:
    def __init__(self, data: float, _parents: Tuple["Value", ...] = (), _op: str = ""):
        self.data: float = data
        self.grad: float = 0
        self._backward: Callable[[], None] = lambda: None
        self._parents: Set["Value"] = set(_parents)
        self._op: str = _op

    # self + other
    def __add__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # self * other
    def __mul__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # self ** other
    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'^{other}')
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def zero_grad(self) -> None:
        self.grad = 0

    def backward(self) -> None:
        topo: List['Value'] = []
        visited: Set['Value'] = set()
        def build_topo(v: 'Value'):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    # -self
    def __neg__(self) -> "Value":
        return self * -1

    # other + self
    def __radd__(self, other: Union["Value", float]) -> "Value":
        return self + other

    # self - other
    def __sub__(self, other: Union["Value", float]) -> "Value":
        return self + (-other)

    # other - self
    def __rsub__(self, other: Union["Value", float]) -> "Value":
        return Value(other) + (-self)

    # other * self
    def __rmul__(self, other: Union["Value", float]) -> "Value":
        return self * other

    # self / other
    def __truediv__(self, other: Union["Value", float]) -> "Value":
        return self * other**-1

    # other / self
    def __rtruediv__(self, other: Union["Value", float]) -> "Value":
        return Value(other) * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
