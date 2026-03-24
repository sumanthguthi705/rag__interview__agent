# Optimization Algorithms

## 1. The Optimization Problem

Training a neural network is an optimization problem: find weights **θ** that minimize the loss function L(θ) over the training data.

```
θ* = argmin_θ  (1/N) Σᵢ L(f(xᵢ; θ), yᵢ)
```

The loss landscape is **non-convex** with millions or billions of dimensions—classical convex optimization guarantees don't apply. Modern deep learning optimizers are **first-order methods** (use gradients, not Hessians) for scalability.

---

## 2. Gradient Descent Variants

### Batch Gradient Descent (Full GD)
```
θ ← θ - η · ∇_θ L(θ)   [using all N training samples]
```
- **Exact gradient**, but too expensive for large datasets.

### Stochastic Gradient Descent (SGD)
```
θ ← θ - η · ∇_θ L(θ; xᵢ, yᵢ)   [one sample at a time]
```
- **Noisy gradient**, but cheap updates. The noise actually helps escape sharp minima.

### Mini-Batch SGD
```
θ ← θ - η · ∇_θ L(θ; Bₜ)   [batch B of size 32-512]
```
- Best of both: vectorized computation, stable gradient estimate, noisy enough to generalize.
- **The standard in practice.**

---

## 3. Momentum

Plain SGD oscillates in ravines (directions of high curvature) and moves slowly along directions of low curvature. **Momentum** accumulates a velocity vector:

```
vₜ = β · vₜ₋₁ + (1-β) · ∇_θ L(θ)   [exponentially decaying average of gradients]
θ ← θ - η · vₜ
```

Typical β = 0.9. Momentum damps oscillations and accelerates in consistent gradient directions, similar to a ball rolling down a hill gaining speed.

### Nesterov Momentum (NAG)
Compute gradient at the **lookahead position** (where momentum would take us):
```
vₜ = β · vₜ₋₁ + η · ∇_θ L(θ - β · vₜ₋₁)
θ ← θ - vₜ
```
Nesterov momentum is more "prescient"—it looks ahead before correcting, leading to faster convergence.

---

## 4. Adaptive Learning Rate Methods

Different parameters may need different effective learning rates. Adaptive methods adjust the step size per-parameter based on historical gradient information.

### AdaGrad (Duchi et al., 2011)
```
G_t += g_t²               (accumulate squared gradients)
θ ← θ - (η / √(G_t + ε)) · g_t
```
- Adapts LR per parameter; larger history → smaller LR.
- **Problem**: G_t only grows → learning rate shrinks to 0, stopping learning prematurely.

### RMSProp (Hinton, 2012)
Fix AdaGrad's monotonic accumulation with exponential moving average:
```
E[g²]_t = ρ · E[g²]_{t-1} + (1-ρ) · g_t²
θ ← θ - (η / √(E[g²]_t + ε)) · g_t
```
- ρ ≈ 0.9 ("forgets" old gradients).
- Effective for RNNs and non-stationary problems.

---

## 5. Adam Optimizer (Kingma & Ba, 2015)

**Adam** (Adaptive Moment Estimation) combines momentum (first moment) with RMSProp (second moment):

```
m_t = β₁ · m_{t-1} + (1-β₁) · g_t          ← First moment (momentum)
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²         ← Second moment (squared gradient)

m̂_t = m_t / (1 - β₁ᵗ)                       ← Bias-corrected
v̂_t = v_t / (1 - β₂ᵗ)                       ← Bias-corrected

θ ← θ - η · m̂_t / (√v̂_t + ε)
```

**Default hyperparameters**: η=0.001, β₁=0.9, β₂=0.999, ε=1e-8

### Why Bias Correction?
At t=1 with β₁=0.9: m₁ = 0.9·0 + 0.1·g₁ = 0.1·g₁ — severely underestimates true gradient. Dividing by (1-β₁¹) = 0.1 corrects this.

### AdamW (Loshchilov & Hutter, 2017)
Standard Adam applies weight decay as L2 regularization (added to gradient). AdamW decouples weight decay from the gradient update, leading to better regularization:

```
θ ← θ - η · (m̂_t / (√v̂_t + ε) + λ·θ)    ← Adam (incorrect decoupling)
θ ← (1 - η·λ)·θ - η · m̂_t / (√v̂_t + ε)  ← AdamW (correct)
```

**AdamW is the standard optimizer for Transformers** (BERT, GPT, LLaMA all use AdamW).

---

## 6. Learning Rate Scheduling

The learning rate η is the most sensitive hyperparameter. A good schedule:
- Starts reasonably high (fast exploration)
- Decays over time (fine-grained convergence)
- Optionally warms up from 0 (avoids instability early)

### Common Schedules

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Step Decay | η *= γ every k epochs | CNNs |
| Cosine Annealing | η_t = η_min + ½(η_max-η_min)(1 + cos(πt/T)) | Transformers |
| Linear Warmup | η_t = η_max · (t/T_warmup) | Transformers |
| One-Cycle | Warmup → peak → annealing | Fast training (SuperConvergence) |
| ReduceLROnPlateau | Decay when val loss stagnates | General purpose |

### Warmup + Cosine (Most Common for Transformers)
```
if t < T_warmup:
    η = η_max * t / T_warmup
else:
    η = cosine_decay(t)
```

---

## 7. The Loss Landscape

### Saddle Points vs. Local Minima
In high-dimensional spaces, **local minima** are rare (requires all dimensions to be locally minimal simultaneously). **Saddle points** (some directions curve up, some down) are far more common and are the real challenge for optimizers.

### Sharp vs. Flat Minima
- **Sharp minima**: Small basin; tiny weight perturbation causes large loss increase. Poor generalization.
- **Flat minima**: Wide basin; robust to perturbations. Better generalization.

Larger batch sizes tend to find sharper minima; smaller batches find flatter ones—partially explaining why large-batch training requires learning rate scaling.

### Stochastic Noise as Regularization
The noise in mini-batch gradient estimates prevents the optimizer from settling into sharp minima, acting as an implicit regularizer.

---

## 8. Interview Key Points

- **Q: Why does Adam converge faster than SGD?**  
  A: Adam adapts learning rates per-parameter, automatically scaling down steps for parameters with large gradients and scaling up for parameters with small gradients. It also uses momentum to accelerate in consistent gradient directions. This makes it much less sensitive to hyperparameter tuning.

- **Q: Why is SGD sometimes preferred over Adam for final training?**  
  A: SGD with momentum often finds flatter minima that generalize better. Adam's adaptive step sizes can overfit to the training set. Many practitioners use Adam for fast initial training, then fine-tune with SGD.

- **Q: What is learning rate warmup and why is it used?**  
  A: Warmup linearly increases the learning rate from 0 to its target over the first few steps. Without warmup, the optimizer makes large parameter updates when gradients are unreliable (early in training), leading to instability. Warmup is especially important for Transformers.

- **Q: What is the difference between AdaGrad and RMSProp?**  
  A: AdaGrad accumulates all past squared gradients monotonically—eventually stopping learning. RMSProp uses an exponential moving average of squared gradients, "forgetting" old gradients and allowing continued learning in non-stationary settings.
