# Bayesian Deep Neural Network for Landslide Susceptibility Mapping

## Overview
This repository presents a **Bayesian Deep Neural Network (BDNN)** implemented in **PyTorch** for binary landslide susceptibility classification.  
Unlike deterministic neural networks, the Bayesian framework provides both **predictive probabilities** and **model uncertainty**, supporting risk-aware geospatial analysis.

---

## Dataset
- Format: Pandas DataFrame  
- Samples: 498  
- Features: 14 landslide conditioning factors  
- Target variable: `Landslide`  
  - `0` → No landslide  
  - `1` → Landslide  
---

## Methodology
- Bayesian Neural Network trained using **Bayes-by-Backprop**
- Variational inference with Gaussian weight distributions
- Monte Carlo sampling for uncertainty estimation
- Binary cross-entropy loss with KL-divergence regularization

---

## Model Outputs
For each sample, the model produces:
- **Landslide probability** (predictive mean)
- **Epistemic uncertainty** (predictive standard deviation)

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC–ROC  
- Confusion Matrix  

Uncertainty quality is assessed through confidence–uncertainty analysis.

---

## Requirements
- Python 3.9 or later  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## Model Serialization
The trained model can be serialized using Python `pickle`.  
For deployment and long-term reproducibility, saving the model `state_dict` along with preprocessing objects is recommended.

---

## Intended Use
- Landslide susceptibility mapping  
- Uncertainty-aware spatial prediction  
- Academic research and experimentation  

This implementation is intended for **research and educational purposes**.

---

## License
Licensed under the **Apache License, Version 2.0**.  
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an **"AS IS" BASIS**, without warranties or conditions of any kind.

See the License for the specific language governing permissions and limitations under the License.

---

## Citation
If you use this work in academic research, please cite the corresponding publication or acknowledge the repository.

