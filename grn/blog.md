# How to Compress a Video into 1s and 0s — Without Losing the Picture

What if you could describe every pixel of a high-resolution video using only a sequence of "yes" or "no" answers, and reconstruct the original *better* than any continuous method?

---

## The Problem

When you train a modern image generator, working directly on raw pixels is brutally expensive. A single 256×256 RGB image contains **196,608 numbers** (256 × 256 × 3). Training a model to predict those one at a time — across billions of images — is impractical.

So researchers use a **VAE (Variational Autoencoder)** as a pre-processing step. Think of it as a very smart image zipper. The VAE encoder squishes your 256×256×3 image down to, say, **16×16×16 = 4,096 numbers** in a "latent space." That's a 48× compression, but unlike regular JPEG, the VAE is trained to preserve the features that matter for reconstruction. The decoder then unzips it back to full resolution.

From now on, your generator only needs to learn the structure of those 4,096 latent numbers — a much more tractable problem.

But here's the catch.

---

## Why It's Hard: The Discrete Tokenization Bottleneck

Language models are extraordinarily good at predicting sequences — one token at a time. The natural move is to use the same approach for images: encode them as a sequence of discrete tokens, then let a transformer predict the next token.

The problem is that VAE latents are **continuous** — they're real-valued floats, not integers. You can't directly feed floats into a language model's vocabulary. You need to map them to a finite set of symbols. That mapping is called **vector quantization (VQ)**.

### How VQ Works (with real numbers)

Let's be concrete. Say your VAE encoder produces a feature map of shape `[C=512, H=8, W=8]` for a single image. Reshape that to `[64, 512]`: you now have **64 spatial positions**, each described by a **512-dimensional vector**.

You pre-define a **codebook** of 1024 representative vectors, each 512-dimensional. For every one of your 64 vectors, you find the closest codebook entry by L2 or cosine distance and replace the vector with that entry's integer index.

Your 64 rich vectors become 64 integers: `[120, 456, 1023, 0, 789, ...]`

**Trivia:** With 1024 codebook entries and 64 positions, there are `1024^64` unique ways to arrange these — a number with 192 digits. That's far more than the number of atoms in the observable universe. In theory, VQ can represent an almost unlimited variety of images. In practice this is why VQ models need huge datasets: the model has to learn which of those astronomically many combinations look like real images.

### Why VQ Breaks Down

The fatal flaw is that the codebook is fixed and coarse. Once you pick entry `120` for a patch, you commit to that exact 512-dimensional vector — no matter how far the true latent was from that entry. If your patch is halfway between two codebook entries, you round to the nearest one and the error is frozen there. There's no way to recover the lost information.

People have tried two fixes:

- **Bigger codebooks**: More entries = finer grid = less error. But bigger vocabularies slow convergence and force you to build larger generators.
- **Lookup-free quantization (FSQ)**: Instead of a learned codebook, each latent dimension is independently rounded to a small set of fixed levels (e.g., `-2, -1, 0, +1, +2`). With 5 levels × 16 dimensions you implicitly get a vocabulary of `5^16 ≈ 150M` — without any lookup table. But you still make one coarse rounding decision per dimension, and whatever error remains, stays.

Both approaches share the same fundamental limit: **one shot, no recovery**.

---

## The Big Idea: Ask Better Questions, Iteratively

**The insight from GRN: instead of one coarse decision, make many fine binary decisions — each one correcting the error left by the last.**

GRN introduces **Hierarchical Binary Quantization (HBQ)**, inspired by how Haar wavelets decompose signals in signal processing. The key mathematical fact: if you're allowed to ask a sequence of yes/no questions about a number, and each question halves the remaining uncertainty, then after M questions the error is bounded by **`1 / 2^M`** — exponentially shrinking.

With M=4 rounds: error < `1/16 = 6.25%`
With M=8 rounds: error < `1/256 = 0.39%` — essentially lossless

And critically: this works without expanding the latent channels at all. You get better quality by adding more *bits*, not more *dimensions*.

---

## How It Works: The HBQ Workflow

Let's trace through the entire pipeline with concrete numbers.

---

### Step 0 — VAE Encoding: Turn Pixels into Latent Floats

A 256×256 RGB video frame passes through the **3D causal VAE encoder**. "Causal" means at inference time, each frame can only attend to *past* frames — the encoder never cheats by looking at the future, which is important for streaming applications.

Architecturally (in `Encoder3d` in `hbq_tokenizer.py`), the encoder:
1. Patchifies the input: a 2×2 spatial patch of 3-channel RGB becomes a 12-channel vector. So spatial resolution halves at the input: 256×256 → 128×128 with 12 channels.
2. Applies several `CausalConv3d` + `ResidualBlock` layers with progressive downsampling — 3 stages of 2× spatial stride and 2 stages of 2× temporal stride.
3. Runs a `ResidualBlock` → `AttentionBlock` → `ResidualBlock` bottleneck.
4. Produces latent `F` of shape `[T, H/16, W/16, C]` — e.g., for a 256×256 image: `[1, 16, 16, 16]` = 4,096 floats.

Each of those 4,096 floats is an unbounded real number, potentially ranging from -100 to +100.

---

### Step 1 — Constrain the Range: tanh

Before HBQ can do its binary bisection, it needs the values in a *bounded* range. So the last thing the encoder does is apply `tanh()`, which squishes every float into `(-1, +1)`.

In code: `HBQ_Tokenizer` sets `encoder_out_type='feature_tanh'`, and `encode()` applies `mu = torch.tanh(mu)` before returning. After this, every latent value is a float in `(-1, +1)`.

**Example**: Say one latent value was `2.3` before tanh. After tanh: `tanh(2.3) ≈ 0.98`. Another was `-0.4`, becomes `tanh(-0.4) ≈ -0.38`. Now we have a bounded domain to work with.

---

### Step 2 — Round 1: The First Binary Split

HBQ starts with center `c₁ = 0` — the midpoint of `(-1, +1)`.

For every element in the latent `F`, ask one question:

> **Is F > c₁?**
> - Yes → `q₁ = 1` (the value lives in the upper half, roughly `(0, 1)`)
> - No  → `q₁ = 0` (the value lives in the lower half, roughly `(-1, 0)`)

**Concrete example**: Our latent value is `F = 0.73`.

- Is `0.73 > 0`? Yes → `q₁ = 1`
- Maximum possible error after this round: `0.5` (we know the value is somewhere in `(0, 1)`, a range of width 1, so the worst-case midpoint error is half that: 0.5)

---

### Step 3 — Update the Center and Ask Again

Now we know the value is in `(0, 1)`. The midpoint of that half is `0.5`. Update the center:

```
c₂ = c₁ + δ(q₁) × 2⁻¹
```

where `δ(1) = +1` and `δ(0) = -1`. So `c₂ = 0 + 1 × 0.5 = 0.5`.

Ask again: **Is `F > c₂`?**

- `0.73 > 0.5`? Yes → `q₂ = 1`
- Now we know the value is in `(0.5, 1)`. Maximum error: `0.25`.

---

### Step 4 — Repeat for M Rounds

Each round, the center moves to the midpoint of the sub-interval the value landed in. Let's run all 4 rounds for `F = 0.73`:

| Round | Center `cᵢ` | Is `0.73 > cᵢ`? | Bit `qᵢ` | New half-range |
|-------|------------|-----------------|----------|----------------|
| 1     | `0.000`    | Yes             | 1        | `(0.0, 1.0)`   |
| 2     | `0.500`    | Yes             | 1        | `(0.5, 1.0)`   |
| 3     | `0.750`    | No              | 0        | `(0.5, 0.75)`  |
| 4     | `0.625`    | Yes             | 1        | `(0.625, 0.75)`|

After 4 rounds: bits = `[1, 1, 0, 1]`. Maximum remaining error: `0.0625` (= `1/16`).

**Analogy**: This is exactly the game "guess a number between 1 and 100, I'll say higher or lower." Each answer halves your remaining search space. After 7 guesses, you've narrowed it to 1 number out of 128 — effectively solved.

---

### Step 5 — Reconstruct the Quantized Feature

From the bits, reconstruct the approximation:

```
F̂ = δ(q₁)×2⁻¹ + δ(q₂)×2⁻² + δ(q₃)×2⁻³ + δ(q₄)×2⁻⁴
```

Plugging in `[1, 1, 0, 1]`:

```
F̂ = (+1)×0.5 + (+1)×0.25 + (-1)×0.125 + (+1)×0.0625
  = 0.5 + 0.25 - 0.125 + 0.0625
  = 0.6875
```

Actual value: `0.73`. Error: `|0.73 - 0.6875| = 0.0425`. Confirmed below the `1/16` bound.

This is exactly **binary fixed-point representation** — the same idea used in your CPU's floating-point unit, except here each bit encodes the direction of the bisection, not a power of 2.

**Why VQ can't do this**: VQ picks from 1024 pre-committed entries in one shot. If your value is `0.73` and the closest codebook entry maps to `0.65`, you're stuck. HBQ has no pre-committed entries — it adapts its buckets dynamically to wherever the value actually lands.

---

### Step 6 — Straight-Through Estimator: Training Through the Hard Part

Here's a problem: the binary decision `q = (F > c)` is a step function. Its gradient is zero almost everywhere — the encoder can't learn from it. This is a deal-breaker for end-to-end training.

The solution is the **Straight-Through Estimator (STE)**:

```
F̂_for_training = StopGrad(F̂ - F) + F
```

In the **forward pass**, the decoder sees the properly quantized `F̂ = 0.6875`. In the **backward pass**, we pretend the quantization didn't happen — gradients flow straight to the encoder's continuous output `F`, as if `F̂ = F`. The encoder learns to produce values that, when quantized, reconstruct well.

This is the same trick used by VQ-VAE, but HBQ benefits more from it: because the quantization error is small by construction (< `1/2^M`), the mismatch between "what the decoder actually got" and "what the gradient thinks it got" is tiny. The STE approximation is highly accurate.

---

### Step 7 — VAE Decoding: Back to Pixels

The quantized feature `F̂` (shape: `[T, H/16, W/16, C]`) goes into the **3D causal VAE decoder** (`Decoder3d` in `hbq_tokenizer.py`). This is the encoder in reverse: progressive upsampling through `CausalConv3d` + `Resample` modules, expanding from the 16×16 latent back to 256×256 pixels.

One engineering detail: the decoder processes video one latent-frame at a time and maintains a **feature cache** — it stores the last 2 timesteps of activations at each layer. This lets causal convolutions see the recent past without re-running the whole encoder, keeping memory usage bounded during long video generation.

---

### Step 8 — Package the Bits for Generation

After tokenizing the full image or video, you have binary labels of shape `[1+T, H/16, W/16, C, M]` — for a 256×256 image with `C=16` channels and `M=4` rounds, that's `1 × 16 × 16 × 16 × 4 = 16,384 bits`.

To feed this to a language model, GRN offers two packing strategies:

**GRN_ind**: Merge the M bits per latent element into a single integer. 4 bits → vocabulary size `2^4 = 16`. Each of the `1 × 16 × 16 × 16 = 4,096` positions becomes one token from a 16-entry vocabulary. Simple and compact.

**GRN_bit**: Keep each bit as its own token (binary vocabulary: `{0, 1}`). The sequence is 4× longer (`16,384` tokens), but each prediction is just a coin flip. This works better for complex video generation: it gives the model more explicit supervisory signal and avoids *token aliasing* (where two different bit patterns produce visually similar images and the model can't distinguish them from the index alone).

---

## The Results

**Image tokenization** (ImageNet 256×256 reconstruction, rFID — lower is better):

| Method | Type | rFID |
|--------|------|------|
| SD-VAE | Continuous (no quantization!) | 0.87 |
| Open-MAGVIT2 | Discrete (18-bit VQ) | 1.17 |
| LlamaGen | Discrete | 2.19 |
| **HBQ (M=4)** | **Discrete** | **0.56** |

HBQ with 4 binary rounds **beats the continuous SD-VAE** at 4× higher compression. The continuous VAE uses 16-bit floats per channel; HBQ represents the same information with 4 bits per channel. That's a factor-of-4 saving in bandwidth, with *better* quality than floating point.

**Video tokenization**: With 8 rounds, the HBQ video tokenizer matches the continuous Wan 2.1 VAE (PSNR 33.98 vs 34.10 dB) at 4× the compression rate. PSNR is measured in decibels — a gap of 0.12 dB is invisible to the human eye. For the first time, a discrete tokenizer is genuinely indistinguishable from its continuous counterpart.

**Why does HBQ beat continuous?** It turns out tanh-bounding the latents and quantizing them hierarchically acts as a form of regularization — it removes high-frequency noise in the latent space that would otherwise confuse the decoder. The quantized `F̂` is actually a *cleaner* signal than the raw encoder output.

---

## Why It Matters

**For researchers and engineers**: You can now build autoregressive image and video models with discrete tokens that are as high-fidelity as diffusion models. Per-token likelihoods become meaningful again — the model can quantify its own uncertainty, skip easy tokens, and focus compute on hard ones.

**For products**: Discrete HBQ tokens and text tokens live in the same world. A single language model backbone can write a caption and generate the corresponding image in one unified sequence — no separate diffusion head required.

**For the field**: HBQ tokens are naturally ordered coarse-to-fine. The first bit of every latent element encodes the broadest semantic structure; the last bit adds fine texture. This means you can generate a rough draft from just the first-round bits, then progressively refine — the same way a human artist sketches before adding detail.

---

## TL;DR

HBQ replaces the one-shot codebook lookup that has always bottlenecked discrete visual tokenizers with a cascading sequence of binary "higher or lower" decisions — each one halving the remaining error — so that after just 4 rounds a discrete tokenizer beats continuous representations, and after 8 rounds it's indistinguishable from them at 4× the compression ratio.
