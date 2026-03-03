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

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
