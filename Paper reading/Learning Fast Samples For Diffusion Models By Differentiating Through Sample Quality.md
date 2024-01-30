## 📖 [LEARNING FAST SAMPLERS FOR DIFFUSION MODELS BY DIFFERENTIATING THROUGH SAMPLE QUALITY](https://arxiv.org/abs/2202.05830)

### GENERALIZED GAUSSIAN DIFFUSION MODELS

$$q_{\mu,\sigma}(\boldsymbol{x}_0,...,\boldsymbol{x}_T)=q(\boldsymbol{x} _0)q(\boldsymbol{x} _T|\boldsymbol{x} _0)\prod _{t=1}^{T-1}q _{\mu,\sigma}(\boldsymbol{x} _t|\boldsymbol{x} _{>t},\boldsymbol{x}_0)$$

$$q_{\mu,\sigma}(\boldsymbol{x} _t|\boldsymbol{x} _{>t},\boldsymbol{x} _0)=\mathcal{N}\left(\boldsymbol{x} _t\bigg|\sum _{u\in S _t}\mu _{tu}\boldsymbol{x} _u,\sigma _t^2\boldsymbol{I}_d\right)$$

where $S_t=\[0,...,T\] / \[1,...,t\]$, and $\mu _{tu}$, $\sigma _t$ are learnable parameters, $\forall t\in\[1,...,T\],u\in S_t$

Using theorem 1. define: $a _{tu}^{(i+1)}=a _{t,t+i}^{(i)}\mu _{t+i,u}+a _{tu}^{(i)}\forall u\in S _{t+i}$ and $\quad v _t^{(i+1)}=v _t^{(i)}+\left(a _{t,t+i}^{(i)}\sigma _{t+i}\right)^2$ letting $a _{tu}^{(1)}=\mu _{tu}\forall u\in S _t$ and $v _t^{(1)} = \sigma _t^2$

Our aim is to get:

$$q _{\mu,\sigma}(\boldsymbol{x} _t|\boldsymbol{x} _0)=\mathcal{N}\left(\boldsymbol{x} _t{\left|a _{t0}^{(T-t+1)}\boldsymbol{x} _0,v _t^{(T-t+1)}\boldsymbol{I} _d\right)}\right.$$

So, let's sample by this(with our learned $\mu$ and $\sigma$, the weight of $x _{>t-1}$ and $\hat{x _{0}}$):

$$p_\theta(\boldsymbol{x} _{t-1}|\boldsymbol{x} _{>t - 1})=q _{\mu,\sigma}\left(\boldsymbol{x} _{t - 1}|\boldsymbol{x} _{>t - 1},\hat{\boldsymbol{x}} _0=\frac1{a _{t0}^{(T-t+1)}}\left(\boldsymbol{x} _t-\sqrt{v _t^{(T-t+1)}}\boldsymbol{\epsilon} _\theta(\boldsymbol{x} _t,t)\right)\right)$$

The accerlate reason is the same as DDIM.
