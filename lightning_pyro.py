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
from pytorch_lightning.loggers import TensorBoardLogger # Import TensorBoard Logger
import torchmetrics

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import pyro.poutine as poutine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
class Config:
    # If file not found, will generate synthetic data
    DATA_PATH = "/content/EDA_data.csv" 
    INPUT_DIM = 0 
    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 30
    NUM_SAMPLES = 100 # MC samples for inference

# -----------------------------
# 2. DATA MODULE
# -----------------------------
class LandslideDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        # Check if file exists, otherwise generate data
        if os.path.exists(self.cfg.DATA_PATH):
            print(f"Loading data from {self.cfg.DATA_PATH}")
            df = pd.read_csv(self.cfg.DATA_PATH)
            X = df.drop(columns=['Landslide']).values
            y = df['Landslide'].values
        else:
            print("Warning: Data file not found. Generating SYNTHETIC data for testing.")
            X, y = make_classification(
                n_samples=1000, n_features=10, n_informative=8, 
                n_classes=2, random_state=42
            )

        # Update input dim dynamically
        self.cfg.INPUT_DIM = X.shape[1]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert to Tensors (y needs to be float for Bernoulli/BCE usually, but Long for some metrics)
        # For Pyro Bernoulli obs, float is preferred.
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

        # Layer 1
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # Layer 2
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
        
        # Logits shape: [batch_size, 1] -> squeeze -> [batch_size]
        logits = self.out(x).squeeze(-1)

        # Deterministic site for prediction extraction
        with pyro.plate("data", x.shape[0]):
            # Probs is tracked for "Predictive", not used in training
            probs = pyro.deterministic("probs", torch.sigmoid(logits))
            
            # Observation
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
        
        # Disable automatic optimization because Pyro SVI handles the step
        self.automatic_optimization = False 

        # Pyro Components
        self.model = BayesianNN(config.INPUT_DIM, config.HIDDEN_DIM)
        # Block 'obs' so the guide doesn't try to predict the target data
        self.guide = AutoNormal(poutine.block(self.model, hide=['obs']))
        
        self.elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, Adam({"lr": config.LR}), loss=self.elbo)

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # SVI Step: Calculates loss, gradients, and updates parameters
        loss = self.svi.step(x, y)
        
        # Convert loss to tensor for logging
        loss_tensor = torch.tensor(loss)
        
        # Log to TensorBoard
        self.log("train_loss", loss_tensor, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_tensor

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Posterior Predictive Sampling
        predictive = Predictive(self.model, guide=self.guide, num_samples=20)
        samples = predictive(x)

        # Mean probability (Monte Carlo integration)
        # samples['probs'] shape: (20, batch_size) -> mean -> (batch_size)
        mean_probs = samples['probs'].mean(dim=0)

        # Reshape to (batch_size, 1) for metrics
        preds = mean_probs.unsqueeze(1)
        targets = y.unsqueeze(1).long() # Metrics often prefer Long targets

        # Calculate Metrics
        self.val_acc(preds, targets)
        self.val_auc(preds, targets)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return None  # Pyro handles optimizers internally

# -----------------------------
# 5. EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Clear Pyro param store
    pyro.clear_param_store()
    pl.seed_everything(42)

    # 1. Setup Data
    conf = Config()
    dm = LandslideDataModule(conf)
    dm.prepare_data()
    dm.setup()
    
    # 2. Setup Model
    # Re-init config with dynamic Input Dim from DataModule
    conf.INPUT_DIM = dm.cfg.INPUT_DIM 
    model = BNNLightning(conf)

    # 3. TensorBoard Logger
    logger = TensorBoardLogger("tb_logs", name="landslide_bnn")

    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(monitor="val_auc", mode="max", filename='best-checkpoint')
    
    trainer = pl.Trainer(
        max_epochs=conf.EPOCHS,
        accelerator="auto", 
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=logger, # Attach Logger
        log_every_n_steps=5,
        enable_progress_bar=True
    )

    print("--- Starting Training (Variational Inference) ---")
    trainer.fit(model, dm)

    # 5. Inference / Uncertainty Estimation
    print("\n--- Running Uncertainty Evaluation on Test Set ---")
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        # Note: We load state_dict into the Pyro param store logic usually, 
        # but PL handles loading the module weights. 
        # For Pyro, params are in the global store, so we just use the trained model.
        pass

    test_loader = dm.test_dataloader()
    X_test, y_test = next(iter(test_loader))

    # Predictive Mode
    predictive = Predictive(model.model, guide=model.guide, num_samples=conf.NUM_SAMPLES)
    samples = predictive(X_test)

    # samples['probs']: (num_samples, batch_size)
    y_pred_prob_samples = samples['probs']

    # 1. Mean Prediction
    y_pred_mean = y_pred_prob_samples.mean(dim=0)
    
    # 2. Uncertainty (Standard Deviation)
    uncertainty = y_pred_prob_samples.std(dim=0)

    # Display Results
    print(f"\n{'True':<5} | {'Pred':<10} | {'Uncertainty':<12} | {'Confidence'}")
    print("-" * 50)

    for i in range(min(15, len(y_test))):
        conf_level = "HIGH" if uncertainty[i] < 0.1 else ("MED" if uncertainty[i] < 0.2 else "LOW")
        print(f"{int(y_test[i]):<5} | {y_pred_mean[i]:.4f}     | {uncertainty[i]:.4f}       | {conf_level}")

    print("\nTo view TensorBoard, run in terminal:")
    print("tensorboard --logdir tb_logs")