## GANS

## Are GANs making a comeback in 2025? 
The [seaweed APT1](https://arxiv.org/pdf/2501.08316) released on 2025 shows a positive direction on how GANs can be used for 1 NFE (a fancy name to call 1 step generation). APT stands for Adverserial post training. In this paper 
- Generator: they first distilled a diffusion model using guidance and then used it is a starting point to train a GAN using raw data keeping T=1 (flow based models). 
- The discriminator is as large as geneator and initialized from a pretrained diffusion model. pick 16, 26, 36 layers which have only single query and does cross attention with k and v. these outputs are concatenated, normalized, and projected to yield a single scalar logit output
- Adds additional loss to discrimintor for R1 regularization
$$
L_{aR1} = \|D(x, c) - D(N(x, \sigma I), c)\|_2^2
$$
- Post evaluation, the authors found that the model outputs more realistic and cinematic photos and videos compared to other distilled diffusion models but their structural integrety and alignment to text took a hit. this more common in videos compared to images. 
