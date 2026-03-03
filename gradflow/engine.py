import math

class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    # self + other
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # self * other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # self ** other
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'^{other}')
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # -self
    def __neg__(self):
        return self * -1

    # other + self
    def __radd__(self, other):
        return self + other

    # self - other
    def __sub__(self, other):
        return self + (-other)

    # other - self
    def __rsub__(self, other):
        return Value(other) + (-self)

    # other * self
    def __rmul__(self, other):
        return self * other

    # self / other
    def __truediv__(self, other):
        return self * other**-1

    # other / self
    def __rtruediv__(self, other):
        return Value(other) * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
