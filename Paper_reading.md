# This .md contains the paper reading from 2024~

## 📖 Understanding Black-box Predictions via Influence Functions

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


