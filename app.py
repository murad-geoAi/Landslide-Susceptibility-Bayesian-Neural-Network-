
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import io

# --- Model Classes (Must match training code) ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.ones(out_features) * -3)

    def forward(self, x):
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return F.linear(x, w, b)

class BayesianDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, 64)
        self.fc2 = BayesianLinear(64, 32)
        self.fc3 = BayesianLinear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# --- Custom Unpickler for CPU Loading ---
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

# --- Feature Statistics for Scaling (Approximated Mean) ---
# Format: 'Feature': {'std': val, 'min': val, 'max': val}
FEATURE_STATS = {
    'Latitude': {'std': 0.3590655196817232, 'min': 21.87579042, 'max': 23.71861474},
    'Longitude': {'std': 0.14315417063756383, 'min': 91.95103668, 'max': 92.6033389},
    'Elevation': {'std': 116.0, 'min': 14.0, 'max': 810.0},
    'Slope': {'std': 7.791320286332246, 'min': 0.0, 'max': 42.56869888},
    'Aspect': {'std': 99.46074895104579, 'min': -1.0, 'max': 357.8789063},
    'Curvature': {'std': 0.8771837422305924, 'min': -4.598128796, 'max': 4.48597908},
    'SPI': {'std': 1.1576537604699766, 'min': -4.089693069, 'max': 4.946649075},
    'TWI': {'std': 1.1576537604699766, 'min': -4.089693069, 'max': 4.946649075},
    'TRI': {'std': 3.5687585579204524, 'min': 0.0, 'max': 22.01009941},
    'NDVI': {'std': 0.08094755220549843, 'min': -0.07335642, 'max': 0.511863887},
    'LULC': {'std': 2.3170172297105385, 'min': 1.0, 'max': 11.0},
    'Annual_Rainfall': {'std': 12.054724113461283, 'min': 22.38495064, 'max': 84.29501343},
    'Drainage_Density': {'std': 0.029953411559536288, 'min': 0.0, 'max': 0.249510005},
    'Soil_Texture': {'std': 1.4491396262708895, 'min': 4.0, 'max': 7.0}
}
# Approximate mean as (min + max) / 2
for key, stats in FEATURE_STATS.items():
    stats['mean'] = (stats['min'] + stats['max']) / 2.0

FEATURES = list(FEATURE_STATS.keys())

# --- Load Model ---
try:
    with open("bayesian_dnn_landslide.pkl", "rb") as f:
        model = CpuUnpickler(f).load()
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Function ---
def predict(*inputs):
    if model is None:
        return "Model not loaded."
    
    # Inputs come as a tuple of values. Order matches FEATURES list.
    processed_inputs = []
    
    for i, feature_name in enumerate(FEATURES):
        val = inputs[i]
        stats = FEATURE_STATS[feature_name]
        # Standard Scaler: (x - mean) / std
        scaled_val = (val - stats['mean']) / stats['std']
        processed_inputs.append(scaled_val)
    
    # Convert to tensor
    x_tensor = torch.tensor([processed_inputs], dtype=torch.float32)
    
    # Predict (Monte Carlo Dropout / averaging is used in notebook, but here we do single pass or detailed?)
    # The notebook used 100 MC samples. Let's do a simple mean of 10 samples for speed/robustness.
    model.eval() # Use eval mode? 
    # Wait, Bayesian layers use randomness in forward()!
    # If we want the mean prediction, we should run multiple times.
    
    predictions = []
    with torch.no_grad():
        for _ in range(20):
            preds = model(x_tensor)
            predictions.append(preds.item())
    
    mean_prob = np.mean(predictions)
    
    # Format output
    risk_level = "High Susceptibility" if mean_prob >= 0.5 else "Low Susceptibility"
    return f"{risk_level} (Probability: {mean_prob:.4f})"

# --- Gradio Interface ---
inputs = []
for feature in FEATURES:
    stats = FEATURE_STATS[feature]
    # Use min/max for slider ranges, but allow typing
    inputs.append(gr.Number(label=feature, value=stats['mean'])) # using mean as default

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Landslide Susceptibility Prediction System",
    description="Enter the geographical features to predict landslide susceptibility. \n\n"
                "**Note**: The model was trained with specific scaling. "
                "This interface approximates the scaler using (min+max)/2 as mean. "
                "For precise results, the original scaler is needed."
)

if __name__ == "__main__":
    iface.launch(share=True)
