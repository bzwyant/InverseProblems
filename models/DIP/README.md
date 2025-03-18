## Introduction

**Deep Image Prior** is a part of the family "untrained networks". It uses randomly-initialized neural network as a handcrafted prior, which shown excellent results in standard inverse problems such as denoising, super-resolution, and inpainting.

## Main Idea

In image restoration problems the goal is to recover original image $x$ having a corrupted image $x_0$. Such problems are often formulated as an optimization task: 
        
$$
\min_x E(x; x_0) + R(x)
$$


where $E(x; x_0)$ is a <i>data term</i> and $R(x)$ is an <i>image prior</i>. The data term $E(x; x_0)$ is usually easy to design for a wide range of problems, such as super-resolution, denoising, inpainting, while image prior $R(x)$ is a challenging one. Today's trend is to capture the prior $R(x)$ with a ConvNet by training it using large number of examples. 


We first notice, that for a surjective $g: \theta \mapsto x$ the following procedure in theory is equivalent to **eq1**:
        
$$\min_\theta E(g(\theta); x_0) + R(g(\theta)) \,.$$

In practice $g$ dramatically changes how the image space is searched by an optimization method. Furthermore, by selecting a "good" (possibly injective) mapping $g$, we could get rid of the prior term. We define $g(\theta)$ as $f_\theta(z)$, where $f$ is a deep ConvNet with parameters $\theta$ and $z$ is a fixed input, leading to the formulation

$$\min_\theta E(f_\theta (z); x_0) \,.$$ 

Here, the network $f_\theta$ is initialized randomly and input $z$ is filled with noise and fixed. 

In other words, <b>instead of searching for the answer in the image space we now search for it in the space of neural network's parameters</b>. We emphasize that we never use a pretrained network or an image database. Only corrupted image $x_0$ is used in the restoration process.  

#### Project page: https://dmitryulyanov.github.io/deep_image_prior
#### Demonstration with image denoising task can be found at: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/denoising.ipynb

## References:

Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9446-9454).