
## Submission
1. `Report`: detailed analysis of our experimental results and findings.
2. `main.ipynb`: the specific code implementation.
   
---
## Dependencies

- torch
- torchvision
- numpy
- tqdm
- pandas
- matplotlib
- seaborn

Install with:

```bash
pip install torch torchvision numpy tqdm pandas matplotlib seaborn
```
---
## Structure of main.ipynb
This notebook systematically compares model fusion strategies and analyzes linear mode connectivity using Fisher information and interpolation curves, revealing connections between initialization, Fisher similarity, and fusion effectiveness.


**1. Dataloader**
- Supports MNIST and CIFAR-10 datasets.
- Normalizes images and returns train, validation, and test DataLoaders.

**2. Models**
- **LeNet5_Caffe**: A simple CNN used for MNIST.
- **ResNet18**: Used for CIFAR-10, loads ImageNet pretrained weights and modifies the final fully connected layer for 10 classes.


**3. Training**
- `set_seed`: Fixes random seeds for reproducibility.
- `get_config`: Randomly selects optimizer (Adam/RMSprop/SGD), learning rate, and number of epochs for each model.
- `train_model`: Trains a single model and saves its weights.
- `evaluate`: Computes test accuracy.
- `train`: Core function that supports both "same initialization" and "different initialization" modes. 

**4. Model Soups**
- **Uniform Soup**: Averages model weights equally.
- **Greedy Soup**: Sorts models by validation accuracy, progressively adds models if they improve validation performance.
- **Ensemble**: Averages output probabilities of multiple models.
- Compares the three fusion methods under different numbers of models and plots performance comparison.

**5. Linear Mode Connectivity**
- **5.1 Fisher Information**
  - `compute_fisher`: Computes the diagonal Fisher information matrix (mean of squared gradients) on the training set.
  - `fisher_cosine_similarity`: Computes cosine similarity between Fisher vectors of two models.
  - Plots a similarity heatmap to analyze representational similarity.

- **5.2 Interpolation Plot**
  - `interpolate_models`: Linearly interpolates between two models with different λ weights.
  - `loss_acc`: Computes validation loss and accuracy for each interpolated model.
  - Computes the **barrier** (maximum excess loss along the interpolation path) to assess whether two models lie in the same loss basin.
  - Plots loss/accuracy curves against λ and marks the barrier.

**6. Visualization and Results Summary**
- Performance comparison tables and plots for model soups.
- Fisher similarity heatmaps.
- Interpolation curves (with barrier) for representative model pairs.

---

Below is the English version of the brief explanation for each section of `main.ipynb`, suitable for your `README.md`:

---

## 1. Dataloader
- Supports MNIST and CIFAR-10 datasets.
- Normalizes images and splits the training set into training/validation (90%/10%).
- Returns train, validation, and test DataLoaders.

---

## 2. Models
- **LeNet5_Caffe**: A simple CNN used for MNIST.
- **ResNet18**: Used for CIFAR-10, loads ImageNet pretrained weights and modifies the final fully connected layer for 10 classes.

---

## 3. Training
- `set_seed`: Fixes random seeds for reproducibility.
- `get_config`: Randomly selects optimizer (Adam/RMSprop/SGD), learning rate, and number of epochs for each model.
- `train_model`: Trains a single model and saves its weights.
- `evaluate`: Computes test accuracy.
- `train`: Core function that supports both "same initialization" and "different initialization" modes. Trains multiple models, records their configs, model instances, and test accuracies.

---

## 4. Model Soups
- **Uniform Soup**: Averages model weights equally.
- **Greedy Soup**: Sorts models by validation accuracy, progressively adds models if they improve validation performance.
- **Ensemble**: Averages output probabilities of multiple models.
- Compares the three fusion methods under different numbers of models and plots performance comparison.

---

## 5. Linear Mode Connectivity

### 5.1 Fisher Information
- `compute_fisher`: Computes the diagonal Fisher information matrix (mean of squared gradients) on the training set.
- `fisher_cosine_similarity`: Computes cosine similarity between Fisher vectors of two models.
- Plots a similarity heatmap to analyze representational similarity.

### 5.2 Interpolation Plot
- `interpolate_models`: Linearly interpolates between two models with different λ weights.
- `loss_acc`: Computes validation loss and accuracy for each interpolated model.
- Computes the **barrier** (maximum excess loss along the interpolation path) to assess whether two models lie in the same loss basin.
- Plots loss/accuracy curves against λ and marks the barrier.

---

## 6. Visualization and Results Summary
- Performance comparison tables and plots for model soups.
- Fisher similarity heatmaps.
- Interpolation curves (with barrier) for representative model pairs.

---

## Summary
This notebook systematically compares model fusion strategies and analyzes linear mode connectivity using Fisher information and interpolation curves, revealing connections between initialization, Fisher similarity, and fusion effectiveness.