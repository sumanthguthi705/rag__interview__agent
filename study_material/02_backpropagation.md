# Backpropagation & Gradients

## 1. What is Backpropagation?

**Backpropagation** (backprop) is the algorithm used to compute the gradient of the loss function with respect to every weight in the network. It applies the **chain rule of calculus** repeatedly, propagating error signals from the output layer back through the network.

Backprop does **not** perform learning itself—it only computes gradients. The actual weight update is done by an **optimizer** (e.g., SGD, Adam).

---

## 2. The Chain Rule

The chain rule allows us to differentiate composite functions:

```
If y = f(g(x)), then dy/dx = (dy/dg) · (dg/dx)
```

In a neural network with layers L, L-1, ..., 1, the gradient of the loss with respect to weights at layer l is computed by chaining partial derivatives:

```
∂L/∂W[l] = ∂L/∂A[L] · ∂A[L]/∂Z[L] · ∂Z[L]/∂A[L-1] · ... · ∂Z[l]/∂W[l]
```

---

## 3. Step-by-Step Backpropagation

### Step 1 — Forward Pass
Compute and cache all pre-activations Z[l] and activations A[l] for each layer.

### Step 2 — Compute Output Gradient (δ at output layer)
For cross-entropy loss with softmax:
```
δ[L] = ∂L/∂Z[L] = A[L] - Y      (elegant closed form)
```

### Step 3 — Backpropagate Through Layers
For each layer l from L-1 down to 1:
```
δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ σ'(Z[l])
```
Where ⊙ is element-wise multiplication and σ'(Z[l]) is the derivative of the activation at layer l.

### Step 4 — Compute Weight Gradients
```
∂L/∂W[l] = δ[l] · A[l-1]ᵀ
∂L/∂b[l] = Σ δ[l]   (sum over batch dimension)
```

### Step 5 — Update Weights (done by optimizer)
```
W[l] ← W[l] - η · ∂L/∂W[l]
```

---

## 4. Computational Graph

Modern frameworks (PyTorch, TensorFlow) represent computation as a **directed acyclic graph (DAG)**. Each node is an operation; edges carry tensors. Backprop traverses this graph in reverse (topological order), accumulating gradients via the chain rule at each node.

**PyTorch Autograd**: `tensor.backward()` triggers reverse-mode automatic differentiation through the dynamic computation graph.

---

## 5. Vanishing Gradient Problem

### Cause
When gradients are backpropagated through many layers, they are multiplied by the derivatives of activation functions. Sigmoid and tanh have derivatives ≤ 0.25 and ≤ 1, respectively. Multiplying many values < 1 together causes the gradient to shrink **exponentially** as it propagates back.

### Effect
Early layers receive near-zero gradients and fail to learn meaningful features. The network effectively learns only from the last few layers.

### Solutions
- **ReLU activations**: Derivative is 1 for positive inputs, preventing gradient shrinkage.
- **Residual (skip) connections**: Provide gradient highways that bypass layers (ResNet).
- **Batch Normalization**: Re-centers activations, keeping them in the non-saturating region.
- **LSTM/GRU gating**: Controls gradient flow through time in sequential models.
- **Gradient clipping**: Caps gradient norms to prevent explosion.

---

## 6. Exploding Gradient Problem

### Cause
When weights are initialized too large or the learning rate is too high, gradients can grow exponentially during backprop, causing weight updates that overflow numerical precision.

### Symptoms
- `NaN` loss values
- Model weights becoming `inf`
- Unstable, oscillating training curves

### Solutions
- **Gradient clipping**: Clip gradient norm to a threshold τ:
  ```
  if ||g|| > τ:  g ← g · (τ / ||g||)
  ```
- **Careful initialization** (Xavier / He)
- **Lower learning rate**
- **Batch normalization**

---

## 7. Gradient Flow in Practice

### Checking Gradient Health
- Plot gradient norms per layer during training.
- Early layers should receive gradients of similar magnitude to later layers.
- Use tools: `torch.nn.utils.clip_grad_norm_()`, gradient histograms in TensorBoard.

### Gradient Accumulation
For large batches that don't fit in memory, accumulate gradients over multiple smaller forward/backward passes before calling `optimizer.step()`.

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 8. Automatic Differentiation vs. Numerical Differentiation

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Numerical (finite diff) | Slow O(n) | Low (floating pt error) | Gradient checking |
| Symbolic | Medium | Exact | CAS tools |
| Automatic (reverse mode) | Fast | Exact | Deep learning training |

**Gradient checking**: Verify backprop correctness by comparing autograd gradients to numerical approximations:
```
(f(x + ε) - f(x - ε)) / (2ε)  ≈  df/dx
```

---

## 9. Interview Key Points

- **Q: What is backpropagation and what does it compute?**  
  A: Backprop is an efficient application of the chain rule to compute exact gradients of the loss w.r.t. all network parameters. It caches forward-pass activations and propagates error signals backward through the computation graph.

- **Q: Why does sigmoid cause vanishing gradients?**  
  A: The sigmoid derivative σ'(z) = σ(z)(1 − σ(z)) ≤ 0.25. Multiplying many such terms together during backpropagation causes the gradient to approach zero exponentially in the number of layers.

- **Q: How does ResNet solve the vanishing gradient problem?**  
  A: Skip connections allow gradients to flow directly from the output back to early layers without passing through every intermediate transformation, providing a "gradient highway."

- **Q: What is the difference between backpropagation and gradient descent?**  
  A: Backprop computes gradients. Gradient descent (or any optimizer) uses those gradients to update weights. They are separate algorithms that work together.
