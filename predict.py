import sys
import argparse
from gradflow.nn.mlp import MLP
from gradflow.tensor import Tensor

def main():
    parser = argparse.ArgumentParser(description="GradFlow Clinical Predictor CLI")
    parser.add_argument("--weights", type=str, default="output_model_weights.json", help="Path to model weights JSON")
    parser.add_argument("--radius", type=float, required=True, help="Mean Radius")
    parser.add_argument("--texture", type=float, required=True, help="Mean Texture")
    parser.add_argument("--smoothness", type=float, required=True, help="Mean Smoothness")
    
    args = parser.parse_args()

    # 1. Load weights and metadata
    # We initialize a dummy MLP first to call load()
    # In a real library, we'd probably have a factory method, but here we'll 
    # stick to the implementation simplicity.
    try:
        # We need the architecture to initialize the MLP
        import json
        with open(args.weights, 'r') as f:
            payload = json.load(f)
        metadata = payload.get('metadata', {})
        
        n_inputs = metadata.get('n_inputs', 3)
        layer_outputs = metadata.get('layer_outputs', [6, 6, 1])
        means = metadata.get('means', [0.0, 0.0, 0.0])
        stds = metadata.get('stds', [1.0, 1.0, 1.0])
        
        model = MLP(n_inputs, layer_outputs)
        model.load(args.weights)
        
    except FileNotFoundError:
        print(f"Error: Weights file '{args.weights}' not found. Run the training in demo.ipynb first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Normalize input
    raw_input = [args.radius, args.texture, args.smoothness]
    normalized_input = [(raw_input[i] - means[i]) / stds[i] for i in range(len(raw_input))]
    
    # 3. Predict
    prediction_tensor = model(Tensor(normalized_input))
    raw_score = prediction_tensor.data[0].data
    
    diagnosis = "MALIGNANT" if raw_score > 0 else "BENIGN"
    confidence = abs(raw_score) # Simple heuristic for confidence in our linear output
    
    print("\n--- GradFlow Diagnosis Report ---")
    print(f"Input Features: Radius={args.radius}, Texture={args.texture}, Smoothness={args.smoothness}")
    print(f"Raw Model Score: {raw_score:.4f}")
    print(f"Diagnosis: {diagnosis}")
    print("---------------------------------\n")

if __name__ == "__main__":
    main()
