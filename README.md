# genai

Different approaches to image and video generation.

## Notebooks

**Score** — training a network to estimate the score function and use it to generate samples

**Diffusion** — forward noising process reversed by a learned denoiser; covers DDPM, DDIM, VAE, VQ-VAE, and DiT

**Flow** — invertible transformations that map noise to data; covers normalizing flows, flow matching, and MeanFlow

**GANs** — generator vs discriminator adversarial training for single-step generation

**VAR** — autoregressive image generation over discrete VQ tokens with a GPT backbone

**GRN** — hierarchical binary quantization that beats continuous VAEs at 4× compression

---

## Setup

```bash
uv sync --extra torch-cpu   # macOS or Linux CPU
uv sync --extra torch-cuda  # Linux with NVIDIA GPU
```
