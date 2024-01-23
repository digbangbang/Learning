# This .md contains the paper reading from 2024~

## 📖 [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)

### Influence function deriving from Appendix A.

Emprical risk: $$R(\theta) \stackrel{\text { def }}{=} \frac{1}{n} \sum L\left(z_i, \theta\right)$$

Emprical Hessian $$H_{\hat{\theta}} \stackrel{\text { def }}{=} \nabla^2 R(\hat{\theta})=\frac{1}{n} \sum \nabla_\theta^2 L\left(z_i, \hat{\theta}\right)$$

upweight training point $z$ to delete $z$ in emprical risk $$\hat{\theta}_{\epsilon, z}=\arg \min _{\theta \in \Theta}\{R(\theta)+\epsilon L(z, \theta)\}$$

define $$\Delta_ \epsilon=\hat{\theta}_ {\epsilon, z}-\hat{\theta}$$

$\hat{\theta}$ is not related to $\epsilon$ $$\frac{d \hat{\theta}_ {\epsilon, z}}{d \epsilon}=\frac{d \Delta_\epsilon}{d \epsilon}$$

first-order optimality condition: $$0=\nabla R\left(\hat{\theta}_ {\epsilon, z}\right)+\epsilon \nabla L\left(z, \hat{\theta}_{\epsilon, z}\right)$$

Talyor expansion $$0\approx\left[\nabla R(\hat{\theta})+\epsilon\nabla L(z,\hat{\theta})\right]+\left[\nabla^2R(\hat{\theta})+\epsilon\nabla^2L(z,\hat{\theta})\right]\Delta_\epsilon$$

$$\Delta_\epsilon\approx-\left[\nabla^2R(\hat{\theta})+\epsilon\nabla^2L(z,\hat{\theta})\right]^{-1}\left[\nabla R(\hat{\theta})+\epsilon\nabla L(z,\hat{\theta})\right]$$

$\nabla R(\hat{\theta})=0$ $$\Delta_\epsilon\approx-\nabla^2R(\hat{\theta})^{-1}\nabla L(z,\hat{\theta})\epsilon$$

finally $$\left.\frac{d\hat{\theta}_ {\epsilon,z}}{d\epsilon}\right|_ {\epsilon=0}=-H_{\hat{\theta}}^{-1}\nabla L(z,\hat{\theta}) \stackrel{\text { def }}{=} \mathcal{I}_{\mathrm{up},\mathrm{params}}(z)$$

### Influence of upweighting $z$ on the loss at a test point $z_{test}$

$$\begin{aligned}
\mathcal{I}_ {\mathrm{up,loss}}(z,z_{\mathrm{test}})& \overset{\mathrm{def}}{=}\left.\frac{dL(z_{\mathrm{test}},\theta_ {\epsilon,z})}{d\epsilon}\right|_ {\epsilon=0}  \\
&=\nabla_\theta L(z_{\mathrm{test}},\hat{\theta})^\top\frac{d\hat{\theta}_ {\epsilon,z}}{d\epsilon}\Big|_ {\epsilon=0} \\
&=-\nabla_\theta L(z_{\mathrm{test}},\hat{\theta})^\top H_{\hat{\theta}}^{-1}\nabla_\theta L(z,\hat{\theta})
\end{aligned}$$

It can be seen from influence function(z) $\frac{d \hat{\theta}_ {\epsilon, z}}{d \epsilon}=\frac{d \Delta_\epsilon}{d \epsilon}$. Apply an influence $\epsilon$ to the data point $z$, the larger the influence function is, the greater the influence $z$ has on $z_{test}$.



## 📖 [DENOISING DIFFUSION IMPLICIT MODELS](https://arxiv.org/abs/2010.02502)

### New inference distribution

DDIM defines a new inference distribution:

$$q _\sigma(\boldsymbol{x} _{1:T}|\boldsymbol{x} _0):=q _\sigma(\boldsymbol{x} _T|\boldsymbol{x} _0)\prod _{t=2}^Tq _\sigma(\boldsymbol{x} _{t-1}|\boldsymbol{x} _t,\boldsymbol{x}_0)$$

where $q_\sigma(\boldsymbol{x}_T|\boldsymbol{x}_0)=\mathcal{N}(\sqrt{\bar{\alpha}_T}\boldsymbol{x}_0,(1-\bar{\alpha}_T)\boldsymbol{I})$

for all $t > 1$
$$q _\sigma(\boldsymbol{x} _{t-1}|\boldsymbol{x} _t,\boldsymbol{x} _0)=\mathcal{N}\left(\boldsymbol{x} _{t-1};\sqrt{\bar{\alpha} _{t-1}}\boldsymbol{x} _0+\sqrt{1-\bar{\alpha} _{t-1}-\sigma_t^2}\cdot\frac{\boldsymbol{x}_t-\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0}{\sqrt{1-\bar{\alpha}_t}},\sigma_t^2\boldsymbol{I}\right)$$

the new distribution ensure $q_\sigma(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0,(1-\bar{\alpha}_t)\boldsymbol{I})$, the prove can be seen in the Appendix B.

just using the property above $\boldsymbol{x}_t=\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$

and we can sample $x_{t-1}$ by using below:
$$\boldsymbol{x} _{t-1}=\sqrt{\bar{\alpha} _{t-1}}\underbrace{\left(\frac{\boldsymbol{x} _{t}-\sqrt{1-\bar{\alpha} _{t}}\epsilon _{\theta}(\boldsymbol{x} _{t}, t)}{\sqrt{\bar{\alpha} _{t}}}\right)} _{\text{“ predicted }\boldsymbol{x} _{0}\text{”}} + \underbrace { \sqrt { 1 - \bar{\alpha}  _ { t - 1 }-\sigma _{t}^{2}}\cdot\epsilon _{\theta}(\boldsymbol{x} _{t}, t)} _{\text{“direction pointing to }\boldsymbol{x} _{t}\text{”}} + \underbrace { \sigma _ { t }\epsilon _{t}} _{\text{random noise}}$$

where $\epsilon_t\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})$, and the author defines $\sigma_t=\eta\cdot\sqrt{(1-\bar{\alpha} _{t-1})/(1-\bar{\alpha} _t)}\sqrt{1-\alpha _t}$

if $\eta = 1$, we get DDPM

$$\boldsymbol{x} _{t-1}={\frac{1}{\sqrt{\alpha _{t}}}}\Big(\boldsymbol{x} _{t}-{\frac{1-\alpha _{t}}{\sqrt{1-\bar{\alpha} _{t}}}}\boldsymbol{\epsilon} _{\theta}(\boldsymbol{x} _{t},t)\Big) + \frac{1-\bar{\alpha} _{t-1}}{1-\bar{\alpha}_t}\(1-\alpha_t)\epsilon _{t}$$

if $\eta = 0$, we get DDIM 

$$\boldsymbol{x} _{t-1}={\frac{1}{\sqrt{\bar{\alpha} _t}}} (\boldsymbol{x} _{t} - \sqrt{1 - \alpha _{t}} \epsilon _{\theta}(\boldsymbol{x} _{t}, t))$$

*DDIM Accelerate explanation*

*The original forward process is obtained by using Markov properties, but the forward process of DDIM is not obtained by using Markov properties.(We use a new distribution to obtain)* -> $q_\sigma(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0,(1-\bar{\alpha}_t)\boldsymbol{I})$

*This shows that in fact, during training, you can not only use 1 as the step size for diffusion, but you can choose a larger step size (actually it has no influence for training, because training can spread in one step, so it is the same as DDPM training).*

Choosing a subset $\[\boldsymbol{x}_{\tau_1},\ldots,\boldsymbol{x} _{\tau_S}\]$ of $1,\ldots,T$

*Then the denoising process will use a larger step size for denoising, thereby obtaining an acceleration effect.*

### code of DDIM sample

```python

if self.args.skip_type == "uniform":
    skip = self.num_timesteps // self.args.timesteps
    seq = range(0, self.num_timesteps, skip)
elif self.args.skip_type == "quad":
    seq = (
        np.linspace(
            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
        )
        ** 2
    )
    seq = [int(s) for s in list(seq)]
else:
    raise NotImplementedError

xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds
```



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



## 📖 [Farewell to Aimless Large-scale Pretraining: Influential Subset Selection for Language Model](https://arxiv.org/abs/2305.12816)

### Influence function deriving by using implicit differentiation (not from this paper) 

Emprical risk: $$R(\theta) \stackrel{\text { def }}{=} \frac{1}{n} \sum L\left(z_i, \theta\right)$$

Emprical Hessian $$H_{\hat{\theta}} \stackrel{\text { def }}{=} \nabla^2 R(\hat{\theta})=\frac{1}{n} \sum \nabla_\theta^2 L\left(z_i, \hat{\theta}\right)$$

upweight training point $z$ to delete $z$ in emprical risk $$\hat{\theta}_{\epsilon, z}=\arg \min _{\theta \in \Theta}\{R(\theta)+\epsilon L(z, \theta)\}$$

influence function is defined by $$\mathcal{I} _{\mathrm{up},\mathrm{params}}(z) \stackrel{\text { def }}{=} \left.\frac{d\hat{\theta} _{\epsilon,z}}{d\epsilon}\right| _{\epsilon=0}$$

first-order optimality condition: $$0=\nabla R\left(\hat{\theta}_ {\epsilon, z}\right)+\epsilon \nabla L\left(z, \hat{\theta}_{\epsilon, z}\right)$$

then define $$f(\epsilon, \hat{\theta} _{\epsilon, z}) = \nabla R\left(\hat{\theta} _{\epsilon, z}\right)+\epsilon \nabla L\left(z, \hat{\theta} _{\epsilon, z}\right) = 0$$

using implicit differentiation $$\frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \epsilon} +  \frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \hat{\theta} _{\epsilon, z}} \cdot \frac{\partial \hat{\theta} _{\epsilon, z}}{\partial \epsilon} = 0$$

so $$\frac{\partial \hat{\theta} _{\epsilon, z}}{\partial \epsilon} = -\Big[\frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \hat{\theta} _{\epsilon, z}}\Big]^{-1} \cdot \frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \epsilon}$$

and we have $$\frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \hat{\theta} _{\epsilon, z}} = \nabla _{\theta} ^{2} R\left(\hat{\theta} _{\epsilon, z}\right)+\epsilon \nabla _{\theta} ^{2} L\left(z, \hat{\theta} _{\epsilon, z}\right)$$

$$\frac{\partial f(\epsilon, \hat{\theta} _{\epsilon, z})}{\partial \epsilon} = \nabla _{\epsilon} \nabla _{\theta} R\left(\hat{\theta} _{\epsilon, z}\right) + \nabla _{\theta} L\left(z, \hat{\theta} _{\epsilon, z}\right)$$

finally $$\frac{\partial \hat{\theta} _{\epsilon, z}}{\partial \epsilon} = -\Big[ \nabla _{\theta} ^{2} R\left(\hat{\theta} _{\epsilon, z}\right)+\epsilon \nabla _{\theta} ^{2} L\left(z, \hat{\theta} _{\epsilon, z}\right) \Big]^{-1} \cdot \Big[\nabla _{\epsilon} \nabla _{\theta} R\left(\hat{\theta} _{\epsilon, z}\right) + \nabla _{\theta} L\left(z, \hat{\theta} _{\epsilon, z}\right)\Big]$$

$$\mathcal{I} _{\mathrm{up},\mathrm{params}}(z) \stackrel{\text { def }}{=} \left.\frac{d\hat{\theta} _{\epsilon,z}}{d\epsilon}\right| _{\epsilon=0} = -H _{\hat{\theta}}^{-1}\nabla L(z,\hat{\theta})$$

### Influence Approximation

Hypothesize that the influences of all training examples (pre-training) $z_p$ on a fixed test point(end-task training) $z_t$ is exactly the total reduction in loss on $z_t$

Minizing $l_p(z_p;\theta,\phi)$ via an iterative optimization procedure (such as $SGD$) which utilizes one training example $z_p$ in iteration $t$. In iteration $t$, $\theta_t$ goes to $\theta_{t+1}$.

The influence of $z_p$ on $z_t$ can be approximated below: $$\mathcal{I}\left(z_{p},z_{t}\right)=l_{t}\left(z_{t},\theta_{t}\right)-l_{t}\left(z_{t},\theta_{t+1}\right)$$

make a first-Taylor expansion of $l_t(z_t,\theta_{t+1})$ at $\theta_t$

$$l_t\left(z_t,\theta_{t+1}\right)= l_{t}\left(z_{t},\theta_{t}\right)+\nabla_{\theta}l_{t}\left(z_{t},\theta_{t}\right)\cdot\left(\theta_{t+1}-\theta_{t}\right) + O\left(\|\theta_{t+1}-\theta_t\|^2\right)$$

using $SGD$ as the optimizer, the update in parameters is $\theta_{t+1} - \theta_{t} = -\eta_t \nabla_{\theta}l_p(z_p,\theta_t)$

disregarding the higher-order, the influence of $z_p$ on $z_t$(below algorithm using validation $z^{\prime}$) is  

$$l_t\left(z_t,\theta_t\right)-l_t\left(z_t,\theta_{t+1}\right)\approx\eta_t\nabla_\theta l_t\left(z_t,\theta_t\right)\cdot\nabla_\theta l_p\left(z_p,\theta_t\right)$$

### Algorithm(from paper)

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/aa8368ab-75b6-44c3-b6f1-d370f7b561a9" alt="legend" width="450" height="600">
</p>

### Experiment reproduction (using [TLM-HYPERPARTISAN](https://huggingface.co/datasets/yxchar/hyp-tlm/tree/main) dataset and [Bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main))

The following code are from [ISS](https://github.com/nitwtog/ISS/tree/main), when using in own machine, I made some changes. First, use the latest pytorch to complete the reproduction.

#### Data Usage

- train.csv -> End-task training dataset
- test.csv -> End-task testing dataset
- dev.csv -> End-task validation dataset

The end-task dataset can also used into pre-training process(in the first training process, it will use train.csv to train the Bert-base-uncased for 15 epoch)

- large_external.csv -> Pre-training corpus(large)
- small_external.csv -> Pre-training corpus(small)

#### Model initialization and Training(Using end-task training dataset)

What we want is to reduce the pre-training corpus(select the important subset), so obviously, it's not suitable to train the model with pre-training corpus(original). First, initializing the model and training with train.csv in 15 epoch or less. 
    
    bash train_score_model.sh

#### Get ISS (使用上面的训练模型，选择使用第几层梯度近似，使用预训练语料库与下游任务训练集计算梯度，以计算ISS)

#### 预训练语料库数据选择

#### 进行下一步训练或者微调


