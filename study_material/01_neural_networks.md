# Neural Networks Fundamentals

## 1. What is a Neural Network?

A neural network is a computational model loosely inspired by the structure of the brain. It consists of **layers of interconnected nodes (neurons)** that transform input data through a series of learned linear and non-linear operations to produce an output.

Neural networks are the foundation of modern deep learning and excel at learning hierarchical representations from raw data—pixels become edges, edges become shapes, shapes become objects.

---

## 2. The Perceptron

The **perceptron** is the simplest neural unit. Given inputs x₁, x₂, ..., xₙ and corresponding weights w₁, w₂, ..., wₙ:

```
output = activation( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
```

Where **b** is the **bias term**, which shifts the activation threshold and is critical for learning non-zero-centered decision boundaries.

A single perceptron can only learn **linearly separable** functions. To learn XOR or any non-linear mapping, multiple perceptrons must be stacked into layers.

---

## 3. Layers in a Neural Network

### Input Layer
Receives raw features. No computation is performed; it simply passes data to the first hidden layer.

### Hidden Layers
Intermediate layers where learned representations live. Each neuron computes:
```
z = Wx + b       (linear transformation)
a = σ(z)         (non-linear activation)
```

### Output Layer
Produces the final prediction. For classification, a **softmax** activation converts raw scores (logits) into a probability distribution over classes.

### Depth vs. Width
- **Depth** (more layers): enables more abstract, hierarchical feature learning.
- **Width** (more neurons per layer): increases the capacity of each representation level.
- The **Universal Approximation Theorem** guarantees that a single hidden layer with enough neurons can approximate any continuous function—but depth makes learning far more efficient in practice.

---

## 4. Activation Functions

Activation functions introduce **non-linearity**, allowing networks to learn complex mappings beyond linear transformations.

### Sigmoid
```
σ(z) = 1 / (1 + e^(-z))   ∈ (0, 1)
```
- Output range (0, 1); interpretable as probability.
- **Problem**: Saturates near 0 or 1, causing **vanishing gradients**. Not used in hidden layers of modern deep networks.

### Tanh (Hyperbolic Tangent)
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))   ∈ (-1, 1)
```
- Zero-centered (better than sigmoid for hidden layers).
- Still suffers from saturation.

### ReLU (Rectified Linear Unit)
```
ReLU(z) = max(0, z)
```
- **Most widely used** in hidden layers. Computationally cheap, does not saturate for positive inputs.
- **Dead ReLU problem**: Neurons can "die" (output 0 permanently) if weights push all inputs below 0.

### Leaky ReLU
```
LeakyReLU(z) = max(αz, z)   where α ≈ 0.01
```
- Fixes dead ReLU by allowing a small gradient for negative inputs.

### GELU (Gaussian Error Linear Unit)
```
GELU(z) ≈ z · Φ(z)
```
- Used in Transformers (BERT, GPT). Smooth approximation to ReLU with stochastic interpretation.

### Softmax
```
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```
- Converts a vector of logits into a probability distribution summing to 1.
- Used in the **output layer** for multi-class classification.

---

## 5. The Forward Pass

During the **forward pass**, data flows from the input layer through each hidden layer to the output layer. At each layer `l`:

```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = activation(Z[l])
```

The final output is compared to the ground truth label using a **loss function**.

---

## 6. Loss Functions

The loss function quantifies how wrong the network's prediction is.

### Mean Squared Error (Regression)
```
MSE = (1/n) Σ (ŷᵢ - yᵢ)²
```

### Binary Cross-Entropy (Binary Classification)
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Categorical Cross-Entropy (Multi-class Classification)
```
CCE = -Σ yᵢ · log(ŷᵢ)
```
Cross-entropy loss penalizes confident wrong predictions heavily due to the log term.

---

## 7. Network Initialization

Poor initialization causes **symmetry breaking failure** or **gradient explosion/vanishing**.

### Zero Initialization (Bad)
All neurons in a layer learn the same features—no differentiation occurs.

### Xavier / Glorot Initialization
For **tanh/sigmoid** activations:
```
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), +√(6/(nᵢₙ + nₒᵤₜ)))
```
Keeps variance constant across layers.

### He (Kaiming) Initialization
For **ReLU** activations:
```
W ~ Normal(0, √(2/nᵢₙ))
```
Accounts for ReLU zeroing half the activations.

---

## 8. Interview Key Points

- **Q: Why do we need non-linear activations?**  
  A: Without them, stacking layers is equivalent to a single linear transformation. Non-linearities enable the network to learn complex, non-linear decision boundaries.

- **Q: What is the role of the bias term?**  
  A: The bias shifts the activation independently of the input, allowing the network to represent functions that do not pass through the origin.

- **Q: Why is ReLU preferred over sigmoid in deep networks?**  
  A: ReLU does not saturate for positive values, producing gradients of 1 and avoiding vanishing gradient issues that make sigmoid-based deep networks hard to train.

- **Q: What does depth buy us compared to width?**  
  A: Depth enables hierarchical feature composition (e.g., edge → texture → object). Theoretically, depth enables exponential compression of some functions that would require exponentially more neurons to represent with a single wide layer.
