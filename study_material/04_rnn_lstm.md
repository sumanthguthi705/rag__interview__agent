# Recurrent Neural Networks & LSTMs

## 1. The Sequence Modeling Problem

Many real-world problems involve **sequential data** where order matters and context depends on history:
- Text, speech, time series, video, music

Standard feedforward networks fail here because:
- They require fixed-size inputs
- They have no mechanism to remember previous time steps
- They cannot model variable-length sequences

**Recurrent Neural Networks (RNNs)** solve this by introducing a **hidden state** that persists across time steps, acting as a compressed memory.

---

## 2. The Vanilla RNN

At each time step t, the RNN updates its hidden state using the current input and the previous hidden state:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

### Key Properties
- **Weight sharing across time**: The same W_hh, W_xh are used at every time step (analogous to CNN weight sharing across space).
- **Variable-length input**: Process sequences of any length.
- **Hidden state h_t**: The "memory" vector that carries information forward.

### Unrolling
An RNN "unrolled" across T time steps looks like a very deep feedforward network of depth T. This is crucial for understanding why training RNNs is hard.

---

## 3. Backpropagation Through Time (BPTT)

Training an RNN requires computing gradients through the unrolled computation graph—a process called **Backpropagation Through Time (BPTT)**.

The gradient of the loss at the final time step T with respect to h_t involves:

```
∂L/∂h_t = (W_hh)^(T-t) · ∂L/∂h_T
```

This matrix power means:
- If eigenvalues of W_hh < 1 → **Vanishing gradients** (can't learn long-range dependencies)
- If eigenvalues of W_hh > 1 → **Exploding gradients** (unstable training)

### Truncated BPTT
To handle very long sequences, truncate backpropagation after k steps. Trades off training efficiency vs. ability to capture long-range dependencies.

---

## 4. Long Short-Term Memory (LSTM)

**LSTM** (Hochreiter & Schmidhuber, 1997) was designed explicitly to solve the vanishing gradient problem with a **cell state** c_t that acts as a long-term memory with controlled read/write/erase operations via learned **gates**.

### The Cell State
The cell state c_t flows through time with only **elementwise multiplication and addition**—operations that preserve gradient magnitudes far better than repeated matrix multiplication.

### The Four Gates

**Forget Gate** — Decide what to erase from cell state:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     ∈ (0, 1)
```

**Input Gate** — Decide what new information to write:
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     ∈ (0, 1)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  ∈ (-1, 1)
```

**Cell Update** — Update the cell state:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t
```
(Forget some old, add some new)

**Output Gate** — Decide what to expose as hidden state:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

### Why LSTMs Work
The **additive cell update** c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t means:
```
∂c_t / ∂c_{t-1} = f_t   (element-wise)
```
As long as the forget gate stays near 1, gradients flow through time without vanishing.

---

## 5. Gated Recurrent Unit (GRU)

The **GRU** (Cho et al., 2014) simplifies the LSTM by merging the forget and input gates into a single **update gate** and eliminating the separate cell state:

```
z_t = σ(W_z · [h_{t-1}, x_t])      ← Update gate
r_t = σ(W_r · [h_{t-1}, x_t])      ← Reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### LSTM vs GRU
| Feature | LSTM | GRU |
|---------|------|-----|
| Parameters | More | Fewer (~75%) |
| Cell state | Separate c_t | Merged |
| Performance | Often slightly better on complex tasks | Faster to train, competitive |
| Best for | Longer sequences, complex tasks | Faster training, similar performance |

---

## 6. Sequence-to-Sequence (Seq2Seq) Models

For tasks like machine translation, a **Seq2Seq** model uses:
- **Encoder RNN**: Reads input sequence, compresses to a fixed context vector h_T (the final hidden state).
- **Decoder RNN**: Generates output sequence conditioned on the context vector.

### Bottleneck Problem
The entire input sequence must be compressed into a single fixed-size vector. For long sequences, this **information bottleneck** causes performance to degrade.

**Solution**: **Attention mechanism** (see Transformers module) allows the decoder to look back at all encoder hidden states dynamically.

---

## 7. Bidirectional RNNs

A standard RNN processes sequences left-to-right. A **Bidirectional RNN** runs one RNN forward and one backward, concatenating their hidden states:

```
h_t = [h_t→, h_t←]
```

This gives each position context from both past and future—critical for tasks like Named Entity Recognition where "Paris" in "Paris Hilton" vs "Paris, France" needs surrounding context.

---

## 8. Common Architectures by Task

| Task | Common Architecture |
|------|---------------------|
| Language modeling | LSTM, Transformer |
| Machine translation | Seq2Seq + Attention, Transformer |
| Sentiment analysis | LSTM, BiLSTM |
| Speech recognition | Bidirectional LSTM + CTC loss |
| Time series forecasting | LSTM, GRU, Temporal Conv Net |

---

## 9. Interview Key Points

- **Q: What is the vanishing gradient problem in RNNs?**  
  A: During BPTT, gradients pass through the matrix W_hh raised to the power of T (sequence length). If eigenvalues of W_hh are < 1, gradients shrink exponentially, making it impossible to learn dependencies beyond a few time steps.

- **Q: How does the LSTM cell state prevent vanishing gradients?**  
  A: The cell state update is additive: c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t. The gradient ∂c_t/∂c_{t-1} = f_t, which can stay close to 1, allowing gradient to flow through arbitrarily long sequences.

- **Q: What is the difference between LSTM and GRU?**  
  A: LSTM has a separate cell state and three gates (forget, input, output). GRU merges forget and input gates into an update gate and has no separate cell state. GRU is simpler and trains faster; LSTM typically performs better on complex long-sequence tasks.

- **Q: Why are Transformers replacing RNNs?**  
  A: RNNs are sequential—each step depends on the previous, preventing parallelization. Transformers process the entire sequence in parallel using self-attention, enabling much faster training on modern hardware and better capture of long-range dependencies.
