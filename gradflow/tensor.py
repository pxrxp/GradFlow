from typing import List, Union, Tuple, Callable, Any
from gradflow.engine import Value

class Tensor:
    # Tensor(data)
    def __init__(self, data: Union[List, float, Value]):
        self.data = data

    # self.data
    @property
    def data(self) -> Union[Value, List]:
        return self._data

    # self.data = value
    @data.setter
    def data(self, value: Union[List, float, Value]):
        self._data = to_value_grid(value)
        self._shape = get_shape(value)

    # self.shape
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    # Examples:
    # 1 -> Value(1)
    # [1, 2, 3] -> [Value(1), Value(2), Value(3)]
    # [[1, 2], [3, 4]] -> [[Value(1), Value(2)], [Value(3), Value(4)]]
    def _to_value_grid(self, data: Union[List, float, Value]) -> Union[Value, List]:
        if isinstance(data, (int, float)):
            return Value(float(data))
        if isinstance(data, list):
            return [self._to_value_grid(d) for d in data]
        if isinstance(data, Value):
            return data
        return data

    # Examples:
    # 1 -> ()
    # [1, 2, 3] -> (3,)
    # [[1, 2], [3, 4]] -> (2, 2)
    # [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> (2, 2, 2)
    def _get_shape(self, data: Union[List, int, float, Value]) -> Tuple[int, ...]:
        if isinstance(data, (int, float, Value)):
            return ()
        if isinstance(data, list):
            return (len(data),) + self._get_shape(data[0])
        return ()

    # self[idx]
    def __getitem__(self, idx: int) -> Union[Value, List]:
        return self.data[idx]

    # self + other
    def __add__(self, other: "Tensor") -> "Tensor":
        assert self.shape == other.shape
        return Tensor(self._op(self.data, other.data, lambda x, y: x + y))

    # self * other
    def __mul__(self, other: Union["Tensor", float, Value]) -> "Tensor":
        if isinstance(other, (int, float, Value)):
            return Tensor(self._map(self.data, lambda x: x * other))
        assert self.shape == other.shape
        return Tensor(self._op(self.data, other.data, lambda x, y: x * y))

    # Apply an operation to two tensors
    def _op(self, a: Union[Value, List], b: Union[Value, List], fn: Callable[[Value, Value], Value]) -> Union[Value, List]:
        if isinstance(a, list) and isinstance(b, list):
            return [self._op(ai, bi, fn) for ai, bi in zip(a, b)]
        if isinstance(a, Value) and isinstance(b, Value):
            return fn(a, b)
        raise ValueError("Shape mismatch")

    # Apply a function to each element of the tensor
    def _map(self, a: Union[Value, List], fn: Callable[[Value], Value]) -> Union[Value, List]:
        if isinstance(a, list):
            return [self._map(ai, fn) for ai in a]
        return fn(a)

    def T(self) -> "Tensor":
        assert len(self.shape) == 2                            # Tensor must be a matrix to transpose
        res = [[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return Tensor(res)

    def matmul(self, other: "Tensor") -> "Tensor":
        assert len(self.shape) == 2 and len(other.shape) == 2  # Tensors must be matrices
        assert self.shape[1] == other.shape[0]                 # Mat1: (m, n), Mat2: (n, p) -> Result: (m, p)
        res = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                dot = Value(0.0)
                for k in range(self.shape[1]):
                    dot += self.data[i][k] * other.data[k][j]
                row.append(dot)
            res.append(row)
        return Tensor(res)

    def sum(self) -> "Tensor":
        flat = self._flatten(self.data)
        res = Value(0.0)
        for x in flat:
            res += x            # This operation is tracked by the Value object
        return Tensor(res)

    def _flatten(self, a: Union[Value, List]) -> List[Value]:
        if isinstance(a, list):
            res: List[Value] = []
            for item in a:
                res.extend(self._flatten(item))
            return res
        return [a]

    def backward(self) -> None:
        flat = self._flatten(self.data)
        assert len(flat) == 1   # Only works for single Value objects
                                # If output is a list of Value objects, 
                                # we'd need Jacobian
        flat[0].backward()

    # To be able to zero out the gradients to prevent accumulation
    def zero_grad(self) -> None:
        for v in self._flatten(self.data):
            v.grad = 0

    # self.grad
    @property
    def grad(self) -> Union[float, List]:
        return self._map_grad(self.data)

    def _map_grad(self, a: Union[Value, List]) -> Union[float, List]:
        if isinstance(a, list):
            return [self._map_grad(ai) for ai in a]
        return a.grad

    # For printing the tensor
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape})"
