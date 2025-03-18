## Introduction

### What are Diffusion Models

In machine learning, diffusion models—also known as diffusion probabilistic models (DPMs) or score-based generative models—are a class of latent variable generative models that learn to model data distributions by simulating a stochastic diffusion process. These models consist of three key components:

1. **Forward Process (Noise Addition)**: The data undergoes a Markovian diffusion process, where Gaussian noise is progressively added over time.
2. **Reverse Process (Denoising)**: A neural network learns to approximate the reverse of this diffusion process to generate new samples.
3. **Sampling Procedure**: Once trained, the model can synthesize new data by reversing the noise perturbation process.

![](https://github.com/user-attachments/assets/b78f5dbb-3920-40ca-8852-03357705bc5b)


### Foundational Models: 


[Denoising Diffusion Probabilistic Models (DDPMs)](https://generativediffusionprior.github.io/)  

Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models that establish a connection between diffusion probabilistic models and denoising score matching using Langevin dynamics. The key idea is to gradually transform a complex data distribution into a simple prior (typically a Gaussian) through a **forward diffusion process**, and then learn to reverse this process to generate new samples.  

#### Forward Process (Diffusion)  

The forward process introduces noise to the data over a sequence of time steps, modeled as a **Markovian process** with Gaussian transitions:  

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$  

where:  
- $x_0 \sim q(x)$ is a sample from the original data distribution.  
- $\beta_t$ is a variance schedule controlling the noise level at step $t$.  
- $x_t$ is obtained by perturbing $x_{t-1}$ with Gaussian noise.  

By iterating this process, the data distribution is gradually transformed into an isotropic Gaussian distribution as $t \to T$. The marginal distribution at any time step can be derived as:  

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I),
$$  

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$.  

#### Reverse Process (Denoising)  

To generate data, we aim to reverse the diffusion process. This is achieved by parameterizing the reverse transition:  

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$  

where a neural network (often a U-Net) is trained to estimate the mean $\mu_\theta(x_t, t)$, often reparameterized as noise prediction $\epsilon_\theta(x_t, t)$. Training is performed using a reweighted variational bound on the negative log-likelihood.  

[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) extends this idea by formulating diffusion models in terms of **Stochastic Differential Equations (SDEs)**. The data distribution is transformed into a known prior via a forward-time SDE:  

$$
dx = f(x, t) dt + g(t) dw,
$$  

where $f(x, t)$ is the drift term, $g(t)$ controls the noise magnitude, and $dw$ is a standard Wiener process. The generative model then learns the corresponding **reverse-time SDE**:  

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{w},
$$  

where \( d\bar{w} \) is a reverse-time Wiener process, and \( \nabla_x \log p_t(x) \) is estimated using a neural network trained via score matching.  

This approach unifies **Score-Based Generative Models (SGMs)** with **DDPMs**, leading to more flexible generative models applicable in a variety of domains, from image synthesis to molecular generation.  


Detailed tutorial can be found [here](https://arxiv.org/pdf/2403.18103).

### Applications: 

#### Video Quality :

[DynamiCrafter](https://doubiiu.github.io/projects/DynamiCrafter/) animates open-domain still images by leveraging pre-trained video diffusion priors, utilizing a dual-stream image injection approach to ensure both semantic understanding and preservation of visual details in the generated videos.

#### Image Restoration:

The [Generative Diffusion Prior (GDP)](https://hojonathanho.github.io/diffusion/) is an unsupervised image restoration framework that leverages pre-trained Denoising Diffusion Probabilistic Models (DDPMs) to address both linear and non-linear degradation issues without requiring known degradation parameters or supervised training. 


#### Medical Imaging

Diffusion models' applications in medical imageing can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001068).

