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
