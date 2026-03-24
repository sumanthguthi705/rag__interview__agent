# Convolutional Neural Networks (CNNs)

## 1. Motivation for CNNs

Fully connected networks applied to images have catastrophic parameter counts. A 224×224 RGB image has 150,528 inputs. A single fully connected layer with 1,000 neurons would require **150 million parameters** in just one layer—before learning anything meaningful.

CNNs exploit two key **inductive biases** of image data:
1. **Translation equivariance**: A cat in the top-left is still a cat in the bottom-right.
2. **Locality**: Nearby pixels are more correlated than distant ones.

These priors allow CNNs to use **shared weights** (filters) that scan spatially, drastically reducing parameters while preserving spatial relationships.

---

## 2. The Convolution Operation

A **filter (kernel)** of size K×K slides across the input feature map with a given **stride** s, computing an element-wise dot product at each position:

```
Output[i,j] = Σₘ Σₙ Input[i·s+m, j·s+n] · Filter[m,n]  +  bias
```

### Output Spatial Dimensions
```
H_out = floor((H_in + 2P - K) / S) + 1
W_out = floor((W_in + 2P - K) / S) + 1
```
Where P = padding, K = kernel size, S = stride.

### Key Parameters
| Parameter | Effect |
|-----------|--------|
| Kernel size (K) | Receptive field per filter |
| Stride (S) | Downsampling rate |
| Padding (P) | Controls output size; "same" padding preserves spatial dims |
| # Filters | Depth of output feature map (# feature detectors) |

---

## 3. Pooling Layers

Pooling reduces spatial dimensions, providing:
- **Translation invariance** (approximate)
- Reduced computation
- Compressed representations

### Max Pooling
Takes the maximum activation in each region. Retains the strongest feature activation. Most commonly used.

### Average Pooling
Takes the mean activation. **Global Average Pooling (GAP)** reduces each feature map to a single scalar—used before classification heads in modern networks (replaces flatten+FC).

---

## 4. Building Blocks of a CNN

```
Input Image
     │
  ┌──▼──────────────────────────────┐
  │  Conv2D (K×K, N filters)        │
  │  → BatchNorm                    │
  │  → ReLU                         │
  │  → (optional MaxPool)           │
  └──────────────────────────────── ┘  × L times
     │
  GlobalAvgPool
     │
  Fully Connected (classifier)
     │
  Softmax Output
```

---

## 5. Receptive Field

The **receptive field** of a neuron is the region of the input image that influences its activation. With:
- K = kernel size
- L = number of convolutional layers (stride=1, no pooling)

```
Receptive Field = L × (K − 1) + 1
```

Deep networks with small kernels (e.g., 3×3) can achieve large receptive fields efficiently. For example, two 3×3 layers have the same receptive field as one 5×5 layer but use fewer parameters (2 × 9 = 18 < 25).

---

## 6. Landmark CNN Architectures

### LeNet-5 (LeCun, 1989)
- First practical CNN; applied to handwritten digit recognition (MNIST).
- 2 conv layers + 3 FC layers; tanh activations.

### AlexNet (Krizhevsky, 2012)
- Won ImageNet 2012 with 15.3% top-5 error vs 26.2% runner-up.
- Key contributions: ReLU activations, dropout, data augmentation, multi-GPU training.

### VGGNet (Simonyan, 2014)
- Very deep (16-19 layers) using only 3×3 convolutions.
- Demonstrated that depth is a critical component of performance.

### GoogLeNet / Inception (Szegedy, 2014)
- **Inception module**: Parallel convolutions at multiple scales (1×1, 3×3, 5×5) concatenated.
- 1×1 convolutions as **bottlenecks** to reduce channel dimensionality cheaply.

### ResNet (He, 2015)
- **Residual connections**: F(x) + x allows training of 100+ layer networks.
- Solves vanishing gradient by providing direct gradient paths.
- Identity mapping ensures adding layers cannot hurt performance.

### DenseNet (Huang, 2016)
- Each layer receives feature maps from **all previous layers** (concatenation).
- Maximum gradient flow and feature reuse; parameter-efficient.

### EfficientNet (Tan & Le, 2019)
- **Compound scaling**: Simultaneously scales depth, width, and resolution with fixed ratios.
- Achieves state-of-the-art accuracy with significantly fewer parameters.

---

## 7. 1×1 Convolutions

A 1×1 convolution applies a learned linear combination across the **channel dimension** without changing spatial dimensions. Uses:
- **Dimensionality reduction**: Reduce channels from 256 → 64 cheaply.
- **Adding non-linearity**: Apply ReLU after 1×1 conv to increase model capacity without spatial cost.

---

## 8. Depthwise Separable Convolutions (MobileNet)

Factor a standard K×K convolution into:
1. **Depthwise conv**: K×K applied independently per channel (no cross-channel mixing).
2. **Pointwise conv**: 1×1 applied across channels.

**Parameter reduction**: For a K×K conv with Cᵢₙ input and Cₒᵤₜ output channels:
- Standard: K² × Cᵢₙ × Cₒᵤₜ
- Depthwise separable: K² × Cᵢₙ + Cᵢₙ × Cₒᵤₜ ≈ **8–9× fewer parameters** (K=3)

---

## 9. Batch Normalization in CNNs

Batch Norm normalizes the activations across the batch dimension **per channel**:
```
BN(x) = γ · (x - μ_B) / (σ_B + ε) + β
```
- μ_B, σ_B: batch mean and std
- γ, β: learned scale and shift parameters
- Benefits: Faster training, higher learning rates, less sensitivity to initialization, mild regularization.

---

## 10. Interview Key Points

- **Q: Why do CNNs use shared weights?**  
  A: Shared weights exploit translation equivariance—the same feature detector (edge, curve) is useful at every location. This reduces parameters from O(H·W·C) to O(K²·C), making deep networks tractable.

- **Q: What is the vanishing gradient problem in very deep CNNs and how does ResNet solve it?**  
  A: In deep CNNs, gradients shrink through many layers. ResNet adds skip connections: output = F(x) + x, so the gradient ∂L/∂x = ∂L/∂output · (∂F/∂x + 1). The "+1" ensures gradients cannot vanish as long as any gradient reaches the output.

- **Q: What is the difference between valid and same padding?**  
  A: "Valid" padding uses no padding, causing the output to shrink. "Same" padding adds zeros to keep output spatial dimensions equal to input dimensions (at stride=1).

- **Q: Why are 3×3 convolutions preferred over larger kernels?**  
  A: Two 3×3 layers have the same effective receptive field as one 5×5 layer but use fewer parameters (18 vs 25) and can apply more non-linearities (one extra ReLU), making them more expressive per parameter.
