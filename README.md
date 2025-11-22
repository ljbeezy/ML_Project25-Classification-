ML_Project25 Classification

## Dependencies
- PyTorch
- NumPy
- Scikit-Learn

## Installation

```
pip install torch numpy scikit-learn
```

## Project Files

### Classification Models
1. **classification(NN).py** - Enhanced Neural Network with regularization
2. **classification(SVM).py** - SVM with hyperparameter tuning
3. **classification(random_forest).py** - Random Forest classifier
4. **adaptive_model_selection.py** - Adaptive model selection framework

## Improvements Made

### 1. Overfitting Reduction
- **Validation Split**: 15-25% validation split per dataset with stratified sampling
- **Dropout Regularization**: 0.2-0.4 dropout rates to prevent co-adaptation
- **Batch Normalization**: Added between layers for stable training
- **L2 Regularization**: Weight decay (1e-4) in optimizer
- **Early Stopping**: Patience of 15 epochs, monitoring validation loss
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients

### 2. Loss Optimization
- **Class Weight Balancing**: Automatic class weight computation to handle imbalanced data
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **AdamW Optimizer**: Better than Adam for weight decay regularization
- **Weighted Cross-Entropy**: Balances loss for imbalanced classes

### 4. Adaptive Model Selection
The **adaptive_model_selection.py** script:
- Trains all three models (NN, SVM, RF) on each dataset
- Evaluates using F1-weighted score on validation set
- Automatically selects the best performing model
- Outputs predictions from the best model
- Saves model comparison scores for analysis

## How to Run

### Individual Models
```bash
python classification(NN).py          # Neural Network
python classification(SVM).py          # SVM
python classification(random_forest).py # Random Forest
```

### Adaptive Selection (Recommended)
```bash
python adaptive_model_selection.py
```

## Output Files

- `ClassificationTestResult_adaptive.txt` - Best model predictions (from adaptive selection)

## Architecture Improvements
- PyTorch NN: 3 hidden layers (128→64→32) with BatchNorm and Dropout
- Automatic feature scaling and missing value imputation
- Stratified K-fold cross-validation for hyperparameter tuning