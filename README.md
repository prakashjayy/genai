# genai

Different approaches to image and video generation.

## Notebooks

**Score**
- `score/00_vector_field.ipynb`
- `score/01_kde.ipynb`
- `score/02_guassian_vector_fields.ipynb`
- `score/03_score_matching_base.ipynb`
- `score/04_connecting_denoising_autoencoders_with_sm.ipynb`

**Diffusion**
- `diffusion/00_transformers.ipynb`
- `diffusion/01_vae.ipynb`
- `diffusion/02_diffusion.ipynb`
- `diffusion/03_vqvae.ipynb`
- `diffusion/04_ddim.ipynb`
- `diffusion/05_dit.ipynb`

**Flow**
- `flow/01_flow.ipynb`
- `flow/02_mean_flow.ipynb`

**GANs**
- `gans/00_gan_loss_understanding.ipynb`

**VAR**
- `var/01_vqgan.ipynb`
- `var/02_label_cond_ar.ipynb`

**GRN**
- `grn/hbq.ipynb`

---

## Setup

```bash
uv sync --extra torch-cpu   # macOS or Linux CPU
uv sync --extra torch-cuda  # Linux with NVIDIA GPU
```
