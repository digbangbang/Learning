## 📖 [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

### Set learning Variance(denoising process) to improve log-likelihood

$v$ is a learnable parameter. In $v$, there is a numerical value in each dimension.

$$\Sigma_\theta(x_t,t)=\exp(v\log\beta_t+(1-v)\log\tilde{\beta}_t)$$

$$q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,\color{red}{\beta_t} \mathbf{I})$$

$$q(x_{t-1}|x_t,x_0)=\mathcal{N}(x _{t-1};\tilde{\boldsymbol{\mu}}(x_t,x_0),\color{red}{\tilde{\boldsymbol{\beta}}_t} \mathbf{I})$$

So, using the objective below as the loss: $$L_{\text{hybrid}} = L _ {\text{simple}} + \lambda L _ {\text{vlb}}$$

$$L_\text{simple}=E _{t,x_0,\epsilon}\left[||\epsilon _t -\epsilon _\theta(x_t,t)||^2\right]$$

$$L _ {\text{vlb}}=\mathbb{E} _{t, x _0,\boldsymbol{\epsilon}}\Big[\frac{(1-\alpha _t)^2}{2\alpha_t(1-\bar{\alpha} _t)\|\boldsymbol{\Sigma} _\theta\|_2^2}\|\boldsymbol{\epsilon} _t-\boldsymbol{\epsilon} _\theta\big(x_t,t\big)\|^2\Big]$$

### Cosine noise schedule

$$\bar{\alpha}_t=\frac{f(t)}{f(0)},\quad f(t)=\cos\left(\frac{t/T+s}{1+s}\cdot\frac\pi2\right)^2$$

### Accelerate 

Sample using an arbitrary subsequence $S$ of $t$ values.

Compute $p(x_{S_{t-1}}|x_{S_{t}})$ as $\mathcal{N}(\mu_\theta(x_{S_t},S_t),\Sigma_\theta(x_{S_t},S_t))$, so it can accelerate the sample process.

*Why it could use the subsequence S of t. Because DDPM is actually a special case of DDIM, DDPM can use shorter step to sample. But the FID may not good enough, and the log-likelihood improved method above can support the shorter step sampling.*

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/628105bf-9454-476d-8703-1b129763b423" alt="legend">
</p>
