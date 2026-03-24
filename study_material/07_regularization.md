# Regularization Techniques

## 1. The Bias-Variance Tradeoff

All regularization techniques address the **bias-variance tradeoff**:

```
Expected Test Error = Bias² + Variance + Irreducible Noise
```

- **High Bias (Underfitting)**: Model too simple, misses true patterns. Error is high on both train and test.
- **High Variance (Overfitting)**: Model too complex, memorizes training data noise. Low train error, high test error.

Regularization **increases bias slightly** to **greatly reduce variance**, improving generalization.

---

## 2. L2 Regularization (Weight Decay)

Add a penalty proportional to the squared magnitude of all weights to the loss:

```
L_reg = L_original + (λ/2) · Σ w²
```

### Effect on Gradient Update
```
∂L_reg/∂w = ∂L/∂w + λ·w
w ← w - η·(∂L/∂w + λ·w) = (1 - η·λ)·w - η·∂L/∂w
```

The factor (1-ηλ) **shrinks weights toward zero** each step—hence "weight decay."

### Why It Works
- Discourages large weights that are sensitive to individual training examples.
- Prefers solutions where many features contribute moderately over solutions where few features contribute hugely.
- Bayesian interpretation: MAP estimation with Gaussian prior on weights.

### Typical Values
λ ∈ [1e-5, 1e-2]. Too large → underfitting. Too small → overfitting.

---

## 3. L1 Regularization (Lasso)

```
L_reg = L_original + λ · Σ |w|
```

### Key Difference from L2
L1 penalty has a constant gradient (±λ), which drives **exact zeros**—producing **sparse weight vectors**. L2 drives weights toward zero but never exactly to zero.

### Use Case
Feature selection: When you want the model to automatically identify which input features are truly informative and zero out others.

### Elastic Net
Combines L1 and L2:
```
L_reg = L + λ₁·Σ|w| + λ₂·Σw²
```

---

## 4. Dropout

**Dropout** (Srivastava et al., 2014) randomly sets a fraction p of neurons to zero during each forward pass in training:

```python
# During training
mask = Bernoulli(1-p)  # each neuron active with prob (1-p)
output = (input * mask) / (1-p)  # scale to maintain expected value

# During inference
output = input  # all neurons active, no scaling needed
```

### Why Dropout Works

1. **Ensemble interpretation**: Each training step trains a different sub-network. At test time, using all neurons approximates averaging over 2^N possible sub-networks.

2. **Co-adaptation prevention**: Neurons cannot rely on specific other neurons always being present—they must learn robust features that work alone.

3. **Noise injection**: Adding noise during training is a classical regularization technique.

### Dropout Rates
- Hidden layers: p ∈ [0.2, 0.5] (drop 20-50% of neurons)
- Input layer: p ≤ 0.2 (don't throw away too much information)
- Transformers: Attention dropout, residual dropout (p ≈ 0.1)

### Important: Scale at Test Time
The division by (1-p) during training (inverted dropout) ensures the expected output magnitude is the same during training and inference—no special handling needed at test time.

---

## 5. Batch Normalization

**Batch Normalization** (Ioffe & Szegedy, 2015) normalizes the activations of each layer:

```
μ_B = (1/m) Σ xᵢ              ← batch mean
σ²_B = (1/m) Σ (xᵢ - μ_B)²   ← batch variance
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)  ← normalize
yᵢ = γ·x̂ᵢ + β                  ← scale and shift (learned)
```

### Benefits
1. **Reduces internal covariate shift**: Stabilizes activation distributions, allowing higher learning rates.
2. **Gradient flow**: Activations stay in non-saturating regions of activation functions.
3. **Mild regularization**: Batch statistics introduce noise, similar to dropout.
4. **Less sensitive to initialization**: Normalization compensates for poor initial weights.

### Training vs. Inference
- **Training**: Normalize using batch statistics μ_B, σ²_B.
- **Inference**: Use running averages of μ and σ² tracked during training (no batch dependency).

### Variants
- **Layer Norm**: Normalizes over feature dimension (Transformers). Batch-size independent.
- **Instance Norm**: Normalizes over spatial dimensions per sample (style transfer).
- **Group Norm**: Normalizes over groups of channels (object detection with small batch).

---

## 6. Data Augmentation

**Data augmentation** artificially expands the training dataset by applying transformations that preserve semantic labels:

### Image Augmentation
| Technique | Example |
|-----------|---------|
| Geometric | Random crop, flip, rotation, shear |
| Color | Brightness, contrast, hue, saturation jitter |
| Noise | Gaussian noise, cutout (random erasure) |
| Mix | Mixup (blend two images), CutMix (paste patches) |
| Auto | AutoAugment (learned augmentation policies) |

### Mixup
```
x̃ = λ·xᵢ + (1-λ)·xⱼ
ỹ = λ·yᵢ + (1-λ)·yⱼ
```
Trains on convex combinations of training pairs. Encourages linear behavior between classes.

### Text Augmentation
- Synonym replacement, random insertion/deletion/swap
- Back-translation (translate to another language and back)
- Paraphrase generation

---

## 7. Early Stopping

Monitor validation loss during training; stop when it begins to increase:

```
best_val_loss = ∞
patience = 10
no_improve = 0

for epoch in training:
    val_loss = evaluate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= patience:
        restore_best_checkpoint()
        break
```

Early stopping implicitly limits model complexity—stopping when the model starts to overfit. It's essentially regularization via training time.

---

## 8. Label Smoothing

Instead of hard one-hot labels (0 or 1), use soft labels:

```
y_smooth = (1-ε)·y_one_hot + ε/K
```

Where K = number of classes, ε ≈ 0.1.

### Benefits
- Prevents the model from becoming overconfidently wrong.
- Improves calibration (predicted probabilities better match true likelihoods).
- Mild regularization effect.

---

## 9. Choosing Regularization Strategies

| Scenario | Recommended Regularization |
|----------|---------------------------|
| Small dataset | Heavy dropout, L2, data augmentation |
| Transformer training | Weight decay (AdamW), dropout, label smoothing |
| CNNs | Data augmentation, batch norm, weight decay |
| Very deep networks | Residual connections + batch norm |
| Overfitting risk | Early stopping + dropout |

---

## 10. Interview Key Points

- **Q: What is the difference between L1 and L2 regularization?**  
  A: L2 (weight decay) penalizes the square of weights, driving them toward zero but never exactly to zero. L1 penalizes the absolute value, producing sparse solutions where many weights are exactly zero. L1 performs implicit feature selection; L2 distributes the shrinkage across all features.

- **Q: How does dropout prevent overfitting?**  
  A: Dropout forces neurons to learn redundant representations that work independently, since they cannot rely on co-occurring neurons. It approximately trains an ensemble of 2^N sub-networks and averages them at inference time.

- **Q: Why is Batch Normalization useful in deep networks?**  
  A: BN normalizes intermediate activations, preventing internal covariate shift (the distribution of activations changing as weights update). This allows using higher learning rates, makes training less sensitive to initialization, and provides mild regularization.

- **Q: Can you combine multiple regularization techniques?**  
  A: Yes, and this is common. E.g., CNNs: BatchNorm + Dropout + Data Augmentation + Weight Decay. Transformers: LayerNorm + Dropout + Label Smoothing + AdamW. The key is avoiding over-regularization (underfitting), which requires tuning on a validation set.
