
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import io

# Define classes as they appear in the notebook
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

    def kl_loss(self):
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        kl_w = -0.5 * torch.sum(
            1 + torch.log(w_sigma**2) - self.w_mu**2 - w_sigma**2
        )
        kl_b = -0.5 * torch.sum(
            1 + torch.log(b_sigma**2) - self.b_mu**2 - b_sigma**2
        )

        return kl_w + kl_b

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

    def kl_loss(self):
        return (
            self.fc1.kl_loss() +
            self.fc2.kl_loss() +
            self.fc3.kl_loss()
        )

class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

try:
    with open("bayesian_dnn_landslide.pkl", "rb") as f:
        content = CpuUnpickler(f).load()
    
    print(f"Type of content: {type(content)}")
    if isinstance(content, dict):
        print(f"Keys: {content.keys()}")
    elif hasattr(content, '__dict__'):
        print("Object with __dict__")
        # print first layer weights shape if possible
        if hasattr(content, 'fc1'):
            print(f"fc1 keys: {content.fc1.w_mu.shape}")
        
    else:
        print(content)

except Exception as e:
    print(f"Error loading pickle: {e}")
