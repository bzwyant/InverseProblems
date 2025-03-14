## Introduction

### What are Diffusion Models

In machine learning, diffusion models, also known as diffusion probabilistic models or score-based generative models, are a class of latent variable generative models. A diffusion model consists of three major components: the forward process, the reverse process, and the sampling procedure. The goal of diffusion models is to learn a diffusion process for a given dataset, such that the process can generate new elements that are distributed similarly as the original dataset. A diffusion model models data as generated by a diffusion process, whereby a new datum performs a random walk with drift through the space of all possible data. A trained diffusion model can be sampled in many ways, with different efficiency and quality.

![](images/diffusion.png)

## Foundational Models: 

[Denoising Diffusion Probabilistic Models (DDPMs)](https://generativediffusionprior.github.io/) framework connected diffusion probabilistic models and denoising score matching with Langevin dynamics, and it naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding.

[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) use SDEs to progressively add noise to data, transforming it into a known prior distribution, and a corresponding reverse-time SDE that leverages neural network-estimated score functions to iteratively denoise and reconstruct the original data.

Detailed tutorial can be found [here](https://arxiv.org/pdf/2403.18103).

## Applications: 

### Video Quality :

[DynamiCrafter](https://doubiiu.github.io/projects/DynamiCrafter/) animates open-domain still images by leveraging pre-trained video diffusion priors, utilizing a dual-stream image injection approach to ensure both semantic understanding and preservation of visual details in the generated videos.

### Image Restoration:

The [Generative Diffusion Prior (GDP)](https://hojonathanho.github.io/diffusion/) is an unsupervised image restoration framework that leverages pre-trained Denoising Diffusion Probabilistic Models (DDPMs) to address both linear and non-linear degradation issues without requiring known degradation parameters or supervised training. 


### Medical Imaging

Diffusion models' applications in medical imageing can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001068).

