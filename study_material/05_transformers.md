# Transformers & Self-Attention

## 1. The Attention Idea

The **attention mechanism** was introduced in 2015 (Bahdanau et al.) to solve the information bottleneck in Seq2Seq models. Instead of compressing the entire input into one vector, attention allows the decoder to **selectively focus** on different parts of the encoder's hidden states at each decoding step:

```
Attention weight αᵢ ∝ score(decoder_state, encoder_state_i)
Context vector = Σᵢ αᵢ · encoder_state_i
```

This was revolutionary—suddenly NMT systems could "look back" at the source sentence dynamically.

---

## 2. The Transformer Architecture (Vaswani et al., 2017)

The landmark paper **"Attention Is All You Need"** eliminated recurrence entirely, using only attention mechanisms. This enabled full parallelization over sequence positions.

### High-Level Structure

```
Input Embeddings + Positional Encoding
        │
  ┌─────▼──────────────────────┐
  │    Encoder Block × N       │
  │  ┌──────────────────────┐  │
  │  │ Multi-Head Self-Attn │  │
  │  │ + Add & Norm         │  │
  │  │ Feed-Forward Network │  │
  │  │ + Add & Norm         │  │
  │  └──────────────────────┘  │
  └─────────────────────────── ┘
        │
  ┌─────▼──────────────────────┐
  │    Decoder Block × N       │
  │  ┌──────────────────────┐  │
  │  │ Masked Self-Attention│  │
  │  │ Cross-Attention      │  │
  │  │ Feed-Forward Network │  │
  │  └──────────────────────┘  │
  └────────────────────────────┘
        │
  Linear + Softmax → Output tokens
```

---

## 3. Scaled Dot-Product Attention

The core computation in a Transformer is **Scaled Dot-Product Attention**:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

### Query, Key, Value (Q, K, V)
- **Query (Q)**: What am I looking for?
- **Key (K)**: What information do I have?
- **Value (V)**: What content do I return?

All three are learned linear projections of the input:
```
Q = X · Wᵠ,  K = X · Wᴷ,  V = X · Wᵛ
```

### The √dₖ Scaling
Without scaling, the dot products QKᵀ grow large in magnitude when dₖ is large (since the sum of dₖ terms). This pushes softmax into very flat or very peaked regions, hurting gradient flow. Dividing by √dₖ stabilizes training.

### Attention as Soft Retrieval
- softmax(QKᵀ/√dₖ) produces **attention weights** (sum to 1 per query position).
- These weights are a soft, differentiable version of nearest-neighbor lookup.
- The output is a weighted sum of Values.

---

## 4. Multi-Head Attention

Instead of computing attention once, Multi-Head Attention computes it **h times in parallel** with different projection matrices, then concatenates the results:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ

where headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢᵛ)
```

### Why Multiple Heads?
Each head can attend to different aspects of the input simultaneously:
- Head 1: syntactic relationships
- Head 2: semantic similarity
- Head 3: positional proximity
- etc.

Using h=8 heads with dₖ = d_model/h = 64 gives the same total computation as single-head attention at d_model=512.

---

## 5. Positional Encoding

Self-attention is **permutation equivariant**—it treats all positions identically. "The cat sat on the mat" and "Mat the on sat cat the" produce the same attention (ignoring position).

To inject positional information, the original Transformer adds **sinusoidal positional encodings** to the input embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Why Sinusoidal?
- Unique encoding for each position.
- Generalizes to sequence lengths not seen during training.
- **Relative positions** can be computed by linear combinations (PE(pos+k) is a linear function of PE(pos)).

Modern models often use **learned positional embeddings** (BERT) or **Rotary Position Embedding (RoPE)** (LLaMA, GPT-NeoX).

---

## 6. The Feed-Forward Sublayer

After attention, each position is passed independently through a **position-wise feed-forward network**:

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
```

Typically d_ff = 4 × d_model = 2048. This expands the representation, applies non-linearity, then projects back. **GELU** is preferred over ReLU in modern Transformers.

---

## 7. Layer Normalization & Residual Connections

Each Transformer sub-block follows the **Pre-LN** or **Post-LN** pattern:

**Post-LN** (original paper):
```
x = LayerNorm(x + Sublayer(x))
```

**Pre-LN** (more stable in practice):
```
x = x + Sublayer(LayerNorm(x))
```

### Layer Norm vs. Batch Norm
- **Batch Norm**: Normalizes across batch dimension per feature. Requires large batches; problematic for variable-length sequences.
- **Layer Norm**: Normalizes across feature dimension per sample. Batch-size independent; natural for sequences.

---

## 8. Transformer Complexity

| Property | Transformer | RNN |
|----------|-------------|-----|
| Per-layer complexity | O(n²·d) | O(n·d²) |
| Sequential operations | O(1) | O(n) |
| Max path length | O(1) | O(n) |
| Memory | O(n²) | O(n) |

Transformers are slower than RNNs for very short sequences but scale better as sequence length n grows in the regime n < d.

---

## 9. BERT vs GPT — Encoder vs Decoder

### BERT (Encoder-only)
- **Bidirectional**: Attends to all positions in all directions.
- Pre-training: Masked Language Modeling (predict masked tokens).
- Best for: Classification, NER, QA (understanding tasks).

### GPT (Decoder-only)
- **Unidirectional (causal)**: Each token attends only to previous tokens (masked attention).
- Pre-training: Next token prediction (language modeling).
- Best for: Text generation, completion, few-shot learning.

### T5, BART (Encoder-Decoder)
- Full encoder-decoder like the original Transformer.
- Best for: Seq2Seq tasks (translation, summarization).

---

## 10. Interview Key Points

- **Q: Why does self-attention scale by √dₖ?**  
  A: Dot products grow with dimension dₖ (variance ≈ dₖ). Without scaling, softmax saturates into near-one-hot distributions, killing gradient flow. Dividing by √dₖ normalizes the variance to ~1, keeping softmax in the sensitive regime.

- **Q: What does each attention head learn?**  
  A: Different heads specialize. Empirically, some heads track syntactic structure, others semantic similarity, others co-reference, etc. The multi-head design lets the model attend to multiple relationship types simultaneously.

- **Q: Why can't Transformers directly handle long sequences?**  
  A: Self-attention is O(n²) in both time and memory because every pair of positions attends to each other. For n=10,000 tokens, this requires 100M attention weights. Solutions include Longformer (sparse attention), Flash Attention (memory-efficient exact attention), and linear attention approximations.

- **Q: What is the key difference between BERT and GPT?**  
  A: BERT uses bidirectional attention (sees full context) and is trained with masked LM—ideal for understanding. GPT uses causal (left-to-right) attention—ideal for generation. Neither is universally better; the choice depends on the downstream task.
