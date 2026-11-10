# NanoGPT from Scratch

A 6.68M-parameter generative causal transformer built from mathematical foundations in PyTorch, based on the GPT-2 decoder-only architecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYdC19NNvBAV6PXmWlGDgxiLfg1U4Gde?usp=sharing)

---

## Architecture & Implementation Details

* **Model Type:** Decoder-only Causal Language Model
* **Parameters:** ~6.68 Million
* **Attention Mechanism:** Scaled Dot-Product Multi-Head Attention with lower-triangular causal masking to prevent lookahead.
* **Normalization:** Pre-Layer Normalization (Pre-LN) architecture inside residual blocks to ensure stable gradient propagation.
* **Embeddings:** Learned token embeddings combined with learned positional encodings.
* **Training Pipeline:** Custom training loop built with PyTorch incorporating:
  * AdamW optimizer with weight decay
  * Cosine annealing learning rate schedule with warmup
  * Gradient clipping to prevent exploding gradients
  * Evaluated and trained on an NVIDIA T4 GPU in Google Colab

---

## Tech Stack
* Python 3
* PyTorch
* CUDA / Google Colab (NVIDIA T4)

---

## Interactive Notebook
Run and inspect the full architecture, attention weights, and text generation directly in Google Colab:
[Open in Colab](https://colab.research.google.com/drive/1ZYdC19NNvBAV6PXmWlGDgxiLfg1U4Gde?usp=sharing)
