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

When we got 

$$E_{y_i\sim\eta(\mathbf{x} _i)}[l(\mathbf{x} _i,y _i,A _\mathbf{s})]\leq\delta(\lambda^l+\lambda^\mu LC)$$

Using Hoeffding's Inequality to proof at least $1-\gamma$:

$$\left|\frac{1}{n}\sum _{i\in[n]}l(\mathbf{x} _{i},y _{i};A _{\mathbf{s}})-\frac{1}{|\mathbf{s}|}\sum _{j\in\mathbf{s}}l(\mathbf{x} _{j},y _{j};A _{\mathbf{s}})\right|\leq\delta(\lambda^{l}+\lambda^{\mu}LC)+\sqrt{\frac{L^{2}\log(1/\gamma)}{2n}}$$

*Proof*

First, Hoeffding's inequality(Additive), for all $\epsilon > 0$,

$$\Pr(\bar{X}_n\geq\mu+\epsilon)\leq e^{-2n\epsilon^2}$$

That is $$\Pr(\bar{X}_n\leq\mu+\epsilon)\geq 1 - e^{-2n\epsilon^2}$$

And using the zero-training loss on $\mathbf{s}$, we got

$$\left|\frac{1}{n}\sum _{i\in[n]}l(\mathbf{x} _{i},y _{i};A _{\mathbf{s}})-\frac{1}{|\mathbf{s}|}\sum _{j\in\mathbf{s}}l(\mathbf{x} _{j},y _{j};A _{\mathbf{s}})\right| = \frac{1}{n}\sum _{i\in[n]}l(\mathbf{x} _{i},y _{i};A _{\mathbf{s}})$$

Put it in Hoeffding's inequality:

$$\Pr( \frac{1}{n}\sum _{i\in[n]}l(\mathbf{x} _{i},y _{i};A _{\mathbf{s}}) \leq E _{y_i\sim\eta(\mathbf{x} _i)}[l(\mathbf{x} _i,y _i,A _\mathbf{s})] + \epsilon)\geq 1 - e^{-2n\epsilon^2}$$

Replace $e^{-2n\epsilon^2}$ to $\gamma$, we got at least $1-\gamma$:

$$\frac{1}{n}\sum _{i\in[n]}l(\mathbf{x} _{i},y _{i};A _{\mathbf{s}}) \leq E _{y_i\sim\eta(\mathbf{x} _i)}[l(\mathbf{x} _i,y _i,A _\mathbf{s})] + \sqrt{\frac{log(1/\gamma)}{2n}}$$

After some tidying up we can get the final form.

*End*

### What's the $\delta$ means and how to find?

$\delta$: With $S$ as the data selection pool, $N$ as all data points, and all points in $S$ covered with $\delta$ as the radius, $N$ can be fully covered.

There is an interesting phrase in the paper: a provided label does not help the core-set loss unless it decreases the covering radius.

The article proposed a selection method that can ignore some outliers. I did not read it carefully, I'll explain the *k-Center-Greedy*(*Algorithm 1*):

According to the upper bound, we only need to optimize $\delta$. The optimization idea of *Algorithm 1* is to find the shortest distance from the point of $N$ to point S respectively(which means we will get $N$ distances at all), but to cover $N$, we need the maximum value of these $N$ shortest distances to cover $N$.




