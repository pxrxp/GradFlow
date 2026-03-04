import json
from typing import List
from gradflow.tensor import Tensor

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> List[Tensor]:
        return []

    def save(self, path: str, metadata: dict = None) -> None:
        params_data = []
        for p in self.parameters():
            # Flatten the multi-dimensional parameter grid into a 1D list of floats
            params_data.append([v.data for v in p._flatten(p.data)])
        
        payload = {
            'params': params_data,
            'metadata': metadata or {}
        }
        
        with open(path, 'w') as f:
            json.dump(payload, f)
        print(f"Model saved to {path}")

    def load(self, path: str) -> dict:
        with open(path, 'r') as f:
            payload = json.load(f)
        
        params_data = payload['params']
        model_params = self.parameters()
        
        if len(params_data) != len(model_params):
            raise ValueError(f"Parameter count mismatch: model has {len(model_params)}, file has {len(params_data)}")
            
        for p_file, p_model in zip(params_data, model_params):
            flat_model = p_model._flatten(p_model.data)
            if len(p_file) != len(flat_model):
                raise ValueError("Tensor shape mismatch within parameters")
            for val_data, val_node in zip(p_file, flat_model):
                val_node.data = float(val_data)
        
        print(f"Model loaded from {path}")
        return payload.get('metadata', {})
