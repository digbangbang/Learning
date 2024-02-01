## 📖 [ACTIVE LEARNING FOR CONVOLUTIONAL NEURALNETWORKS: A CORE-SET APPROACH](https://arxiv.org/abs/1708.00489)

### LEMMA 1 Proof Add

About the inequality below, $f:\mathbb{R}^n\to\mathbb{R}^m$:

$$\left\||f(\mathbf{x})-f(\mathbf{y})\right\||_2\leq\left\||J\right\||_F^*\left\||\mathbf{x}-\mathbf{y}\right\||_2\mathrm{~}\forall\mathbf{x},\mathbf{y}\in\mathbb{R}^n$$

I have a proof idea in $1d$, $f:\mathbb{R}^1\to\mathbb{R}^1$:

$$\left\|f(\mathbf{x})-f(\mathbf{y})\right\|\leq\left|\nabla f \right|^*\left\|\mathbf{x}-\mathbf{y}\right\|\mathrm{~}\forall\mathbf{x},\mathbf{y}\in\mathbb{R}^1$$

where the $*$ means the maximum of all.

*Proof*

Let's assume the $1d$ situation: $f(\mathbf{x})$ is not always the linear function, otherwise, the above equation will take the equal sign. 

So, there must have at least 1 point above the line connecting $\mathbf{x}$ and $\mathbf{y}$ (the proof method is the same when it is below situation). We can assume the point is $\eta, \mathbf{x} < \eta < \mathbf{y}$. Then we can get the inequality below:

$$\frac{f(\eta)-f(\mathbf{x})}{\eta-\mathbf{x}}>\frac{f(\mathbf{y})-f(\mathbf{x})}{\mathbf{y}-\mathbf{x}}>\frac{f(\mathbf{y})-f(\eta)}{\mathbf{y}-\eta}$$

Then using Lagrange's mean value theorem, existing $\xi_1, \mathbf{x} < \xi_1 < \eta$ and $\xi_2, \eta < \xi_2 < \mathbf{y}$, meeting:

$$\nabla f(\xi_1) = \frac{f(\eta)-f(\mathbf{x})}{\eta-\mathbf{x}}>\frac{f(\mathbf{y})-f(\mathbf{x})}{\mathbf{y}-\mathbf{x}}>\frac{f(\mathbf{y})-f(\eta)}{\mathbf{y}-\eta} = \nabla f(\xi_2)$$

That means:

$$\left\|f(\mathbf{x})-f(\mathbf{y})\right\|\leq\left|\nabla f \right|^*\left\|\mathbf{x}-\mathbf{y}\right\|$$

It can be expanded to multi demensions.

*End*

About the reverse triangle inequality how to proof:

$$|l(\mathbf{x},y;\mathbf{w})-l(\mathbf{\tilde{x}},y;\mathbf{w})|=|\|CNN(\mathbf{x};\mathbf{w})-y\|_2-\|CNN(\mathbf{\tilde{x}};\mathbf{w})-y\|_2|\leq\|CNN(\mathbf{x};\mathbf{w})-CNN(\mathbf{\tilde{x}};\mathbf{w})\|_2$$

By using the inequality below:

$$|x-y|\geq||x|-|y||.$$

*Proof*

$$|x|+|y-x|\geq|x+y-x|=|y|$$

$$|y|+|x-y|\geq|y+x-y|=|x|$$

That means $$|x-y|\geq||x|-|y||$$

*End*

Using the inequality above:

$$|\|CNN(\mathbf{x};\mathbf{w})-y\|_2-\|CNN(\mathbf{\tilde{x}};\mathbf{w})-y\|_2|\leq\|(CNN(\mathbf{x};\mathbf{w}) - y) -(CNN(\mathbf{\tilde{x}};\mathbf{w}) - y)\|_2$$

We can get the target inequality.

### THROREM 1 Proof Add


### What's the $\delta$ means and how to find?


