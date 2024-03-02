## 📖 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

### The Reverse SDE solution

Forward process SDE

$$d\boldsymbol{x}=\boldsymbol{f}_t(\boldsymbol{x})dt+g_td\boldsymbol{w}$$

Reversed process SDE

$$d\boldsymbol{x}=\left[\boldsymbol{f}_t(\boldsymbol{x})-g_t^2\nabla_x\log p_t(\boldsymbol{x})\right]dt+g_td\boldsymbol{w}$$

*Proof*

First, let $\Delta t\to 0$ and replace the forward process SDE

$$x_{t+\Delta t}-x_t=f_t(x_t)\Delta t+g_t\sqrt{\Delta t}\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)$$

The reason why the term of $\varepsilon$ has a $\sqrt{\Delta t}$ is that the Brown motion ($w_{t+s} - w_{s} \sim\mathcal{N}(0,t)$ , so from $x_t$ to $x_{t+\Delta t}$, $d\boldsymbol{w}\sim\mathcal{N}(0,\Delta t)$ )

Next, we get

$$p(\boldsymbol{x} _{t+\Delta t}|\boldsymbol{x} _t) \left.=\mathcal{N}\left(\boldsymbol{x} _{t+\Delta t};\boldsymbol{x} _t+\boldsymbol{f} _t(\boldsymbol{x} _t)\Delta t,g _t^2\Delta t\right.\boldsymbol{I}\right)  \\
\propto\exp\left(-\frac{\|\boldsymbol{x} _{t+\Delta t}-\boldsymbol{x} _t-\boldsymbol{f} _t(\boldsymbol{x} _t)\Delta t\|^2}{2g _t^2\Delta t}\right)$$

Then, we need to get the $p(\boldsymbol{x} _{t}|\boldsymbol{x + \Delta} _t)$, using Bayesian

$$p(\boldsymbol{x} _t|\boldsymbol{x} _{t+\Delta t})=\frac{p(\boldsymbol{x} _{t+\Delta t}|\boldsymbol{x} _t)p(\boldsymbol{x} _t)}{p(\boldsymbol{x} _{t+\Delta t})}=p(\boldsymbol{x} _{t+\Delta t}|\boldsymbol{x} _t)\exp(\log p(\boldsymbol{x} _t)-\log p(\boldsymbol{x} _{t+\Delta t}))$$

$$\propto\exp\left(-\frac{\|\boldsymbol{x} _{t+\Delta t}-\boldsymbol{x} _t-\boldsymbol{f} _t(\boldsymbol{x} _t)\Delta t\|^2}{2\boldsymbol{g} _t^2\Delta t}+\log p(\boldsymbol{x} _t)-\log p(\boldsymbol{x} _{t+\Delta t})\right)$$

Using Taylor 

$$\log p(x_{t+\Delta t})\approx\log p(x_t)+(x_{t+\Delta t}-x_t)\cdot\nabla_{x_t}\log p(x_t)+\Delta t\frac\partial{\partial t}{\log p(x_t)}$$

Put the Taylor into the function above, then get

$$p(\boldsymbol{x} _t|\boldsymbol{x} _{t+\Delta t})\propto\exp\left(-\frac{\|\boldsymbol{x} _{t+\Delta t}-\boldsymbol{x} _t-\left[\boldsymbol{f} _t(\boldsymbol{x} _t)-g _t^2\nabla _{\boldsymbol{x} _t}\log  p(\boldsymbol{x} _t)\right]\Delta t\|^2}{2g _t^2\Delta t}+\mathscr{O}(\Delta t)\right)$$

Omit the $\mathscr{O}(\Delta t)$

$$\begin{aligned}
p(\boldsymbol{x} _t|\boldsymbol{x} _{t+\Delta t})& \propto\exp\left(-\frac{\|\boldsymbol{x} _{t+\Delta t}-\boldsymbol{x} _t-\left[\boldsymbol{f} _t\left(\boldsymbol{x} _t\right)-g _t^2\nabla _{\boldsymbol{x} _t}\log  p(\boldsymbol{x} _t)\right]\Delta t\|^2}{2g _t^2\Delta t}\right)  \\
&\approx\exp\left(-\frac{\|x _t-x _{t+\Delta t}+\left[f _{t+\Delta t}(x _{t+\Delta t})-g _{t+\Delta t}^2\nabla _{\boldsymbol{x} _{t+\Delta t}}\log  p(\boldsymbol{x} _{t+\Delta t})\right]\Delta t\|^2}{2g _{t+\Delta t}^2\Delta t}\right)
\end{aligned}$$

That means $$x _t\sim \mathcal{N}(x _t; x _{t+\Delta t}-[f _{t+\Delta t}(x _{t+\Delta t})-g _{t+\Delta t}^2\nabla _{\boldsymbol{x} _{t+\Delta t}}\log p(\boldsymbol{x} _{t+\Delta t})]\Delta t, g _{t+\Delta t}^2\Delta t)$$

Finally we get $d\boldsymbol{x}=\left[\boldsymbol{f}_t(\boldsymbol{x})-g_t^2\nabla_x\log p_t(\boldsymbol{x})\right]dt+g_td\boldsymbol{w}$

*End*


Come from 苏剑林


