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

**Let's see the INFO.log**

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/28e355d8-e387-4abc-b8e5-6354d7240f7d" alt="legend" width="600" height="400">
</p>

*It seems the test_loss is calculated wrong, but doesn't matter. Maybe the model is not perfect and leaving some spalce to optimize, let's ignore this and go on.*

#### Get ISS (using small_external.csv for example)

Changing some parameters in both documents, then begin select subset from small_external.csv.

    bash get_score.sh
    
#### Selct topK

    python select_data_byscore_for.py

#### Pretraining

Use the filtered unsupervised dataset and the downstream task dataset to perform the Bert model's pretraining tasks

    bash train.sh

**Let's see the log**

I choose the iterations 50000 to 60000, and the best test macro_f1 is 0.9358

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/e85f967a-d249-47ff-8c89-27cfe7fe3722" alt="legend" width="1000" height="200">
</p>
