import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
class Config:
    DATA_PATH = "EDA_data.csv"
    INPUT_DIM = 0 # Will be set dynamically
    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 50
    NUM_SAMPLES = 100 # Monte Carlo samples for prediction

# -----------------------------
# 2. DATA MODULE
# -----------------------------
class LandslideDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.DATA_PATH)
        X = df.drop(columns=['Landslide']).values
        y = df['Landslide'].values

        # Update input dim based on data
        self.cfg.INPUT_DIM = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert to Tensors
        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        self.test_data = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.cfg.BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.cfg.BATCH_SIZE)

# -----------------------------
# 3. PYRO MODEL
# -----------------------------
class BayesianNN(PyroModule):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # Layer 1: Bayesian Linear
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # Layer 2: Bayesian Linear
        self.fc2 = PyroModule[nn.Linear](hidden_dim, hidden_dim // 2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim // 2, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim // 2]).to_event(1))

        # Output Layer
        self.out = PyroModule[nn.Linear](hidden_dim // 2, 1)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim // 2]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Forward pass gives logits
        logits = self.out(x).squeeze(-1)
        
        # Track actual probability for prediction (deterministic site)
        # This is crucial for retrieving the 'p' values during inference
        probs = pyro.deterministic("probs", torch.sigmoid(logits))

        # Sampling (Observation)
        with pyro.plate("data", x.size(0)):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
            
        return logits

# -----------------------------
# 4. LIGHTNING MODULE
# -----------------------------
class BNNLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        # Pyro Components
        self.model = BayesianNN(config.INPUT_DIM, config.HIDDEN_DIM)
        # AutoNormal approximates the posterior as a Normal distribution
        self.guide = AutoNormal(self.model)
        
        # Loss & Optimizer (Pyro specific)
        self.elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, Adam({"lr": config.LR}), loss=self.elbo)

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        # SVI.step performs the forward pass, calculates loss, and updates params
        # Note: We return the loss value for logging, but PL doesn't do the backprop here.
        loss = self.svi.step(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return torch.tensor(loss) # Return tensor for PL compatibility

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Use Predictive to sample from the Guide (Posterior)
        # We capture "probs" (deterministic site) and "obs" (sampled classes)
        predictive = Predictive(self.model, guide=self.guide, num_samples=20)
        samples = predictive(x)
        
        # Average probability across MC samples
        mean_probs = samples['probs'].mean(dim=0)
        
        # Calculate metrics
        self.val_acc(mean_probs, y)
        self.val_auc(mean_probs, y)
        
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_auc", self.val_auc, prog_bar=True)

    def configure_optimizers(self):
        # Important: Pyro manages its own optimizer inside SVI.
        # We return None to tell Lightning not to manage optimizers.
        return None 

# -----------------------------
# 5. EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Clear Pyro param store to prevent overlap in interactive sessions
    pyro.clear_param_store()
    pl.seed_everything(42)

    # 1. Setup Data
    conf = Config()
    dm = LandslideDataModule(conf)
    dm.prepare_data() 
    dm.setup()
    
    # 2. Setup Model
    model = BNNLightning(conf)

    # 3. Trainer
    checkpoint_callback = ModelCheckpoint(monitor="val_auc", mode="max")
    trainer = pl.Trainer(
        max_epochs=conf.EPOCHS,
        accelerator="auto", # auto-detect GPU/CPU
        devices="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    print("--- Starting Training (Variational Inference) ---")
    trainer.fit(model, dm)

    # 4. Uncertainty Estimation (Inference)
    print("\n--- Running Uncertainty Evaluation ---")
    
    # Get test data
    test_loader = dm.test_dataloader()
    X_test, y_test = next(iter(test_loader)) # Get one batch for demo
    
    # We use the trained guide to make predictions
    predictive = Predictive(model.model, guide=model.guide, num_samples=conf.NUM_SAMPLES)
    samples = predictive(X_test)
    
    # Extract Probabilities
    # samples['probs'] shape: (num_samples, batch_size)
    y_pred_prob_samples = samples['probs']
    
    # Mean Prediction (Point Estimate)
    y_pred_mean = y_pred_prob_samples.mean(dim=0)
    
    # Uncertainty (Standard Deviation of the posterior predictive)
    uncertainty = y_pred_prob_samples.std(dim=0)

    # Metrics on this batch
    preds_binary = (y_pred_mean > 0.5).float()
    acc = torchmetrics.functional.accuracy(preds_binary, y_test, task="binary")
    
    print(f"Test Batch Accuracy: {acc:.4f}")
    
    # Show Uncertainty Examples
    print("\nExamples of Uncertainty:")
    print(f"{'True Label':<10} | {'Pred Prob':<10} | {'Uncertainty (StdDev)':<20} | {'Confidence'}")
    print("-" * 65)
    
    for i in range(10):
        confidence = "HIGH" if uncertainty[i] < 0.1 else ("MED" if uncertainty[i] < 0.2 else "LOW")
        print(f"{int(y_test[i]):<10} | {y_pred_mean[i]:.4f}     | {uncertainty[i]:.4f}               | {confidence}")