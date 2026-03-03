from typing import List, Union, Tuple
from gradflow.engine import Value

class Tensor:
    def __init__(self, data: Union[List, float, Value]):
        self.data: Union[Value, List] = self._to_value_grid(data)
        self.shape: Tuple[int, ...] = self._get_shape(data)

    def _to_value_grid(self, data: Union[List, float, Value]) -> Union[Value, List]:
        if isinstance(data, (int, float)):
            return Value(float(data))
        if isinstance(data, list):
            return [self._to_value_grid(d) for d in data]
        return data

    def _get_shape(self, data: Union[List, float, Value]) -> Tuple[int, ...]:
        if isinstance(data, (int, float, Value)):
            return ()
        if isinstance(data, list):
            return (len(data),) + self._get_shape(data[0])
        return ()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape})"
