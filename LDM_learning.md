# Latent-diffusion Annotation (code in [latent-diffusion](https://github.com/CompVis/latent-diffusion))

## [Autoencoder](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py)

### Calculation formula of kl divergence and negative log likelihood of normal distribution: [DiagonalGaussianDistribution](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/distributions/distributions.py#L24)

After processing the image into a hidden variable z, it is divided into two parts according to the channel dimension, one is mean and the other is logvar.

This code includes the calculation of KL Divergence, Negative log-likelihood

#### KL Divergence Calculation

The KL divergence between two diagonal Gaussian distributions is calculated as follows:
  
  $$D_{KL}(N(\mu_1, \sigma_1^2) || N(\mu_2, \sigma_2^2)) = \frac{1}{2} \sum (\sigma_1^2 + (\mu_1 - \mu_2)^2) / \sigma_2^2 - 1 + \log(\sigma_2^2 / \sigma_1^2)$$

  where $\mu_1$, $\sigma_1^2$ are the mean and variance of the current distribution, and $\mu_2$, $\sigma_2^2$ are the mean and variance of the other distribution.

#### Negative log-likelihood Calculation

The Negative log-likelihood is calculated as follows:

  $$NLL = \frac{n}{2} \log(2\pi) + \frac{n}{2} \log(\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2$$

  where $\mu$ is the sample mean, $\sigma$ is the variance of the current distribution.

### Using GAN to train the Autoencoder: [LPIPSWithDiscriminator](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py#L7)

This is the loss between `input` and `input -> encode -> sample -> decode`, using GAN to train the Encoder and the Decoder.

$$L_{\text {Autoencoder }}=\min_{\mathcal{E}, \mathcal{D}} \max_\psi\left(L_{\text {rec }}(x, \mathcal{D}(\mathcal{E}(x)))-L_{a d v}(\mathcal{D}(\mathcal{E}(x)))+\log D_\psi(x)+L_{r e g}(x ; \mathcal{E}, \mathcal{D})\right)$$

#### $L_{\text {rec }}(x, \mathcal{D}(\mathcal{E}(x)))$

`rec_loss` includes reconstructions' loss and perceptual losss

#### $-L_{a d v}(\mathcal{D}(\mathcal{E}(x)))$

`g_loss` represents the $L_{a d v}$

#### $\log D_\psi(x)$

`logits_fake` is the evaluation to the reconstructions by the disctiminator

#### $L_{r e g}(x ; \mathcal{E}, \mathcal{D})$

`kl_loss` is used to regularize the weight

Using the loss above and the Discriminator, then training the Encoder and Decoder. Finally, we get the Encoder and Decoder(Autoencoder):

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/411c682c-2a97-438f-96c4-9c2423280834" alt="encoder and decoder">
</p>

## [UNet](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L413)

In [text2image](https://github.com/CompVis/latent-diffusion/blob/main/configs/latent-diffusion/txt2img-1p4B-eval.yaml), this UNet could process 3 types of datainput:

- x (image)
- t (timestep)
- context (text_encode)

This defines a class that facilitates processing of multiple data forms:

```python
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
```

When the layerinput is the type of TimestepBlock, it will process the `x` and the `t`. If the layerinput is the type of SpatialTransformer, it will process the `x` and the `context`. Otherwise, it will only process the `x`.

**Let's talk the 3 layers used in UNet that process the different type data:**

### [ResBlock](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L163)

The ResBlock is used to process the `x` and the `t` and this block can change the channels by upsample and downsample.

In the UNet structure, when it is the downsampling process, there are two type ResBlock. One is that only change the channel, other is that also change the size.

Downsample uses the convolution or average pooling to change the size. Upsample uses the interpolate to revoer the size.

### [AttentionBlock](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L278C7-L278C21)

The AttentionBlock is used to process only the `x`, and only used when it is need in [attention_resolutions](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/txt2img-1p4B-eval.yaml#L27).

### [SpatialTransformer](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L218)

The SpatialTransformer is used to process the `x` and the `context`, and the crossattention is added into the SpatialTransformer.

In the crossattention, the Q represents the `x`, the K and the V represents the `context`. If there is no `context`, then it will become the self-attention.


<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/f405de99-241d-41cb-a767-ec2969b1a971" alt="UNet">
</p>

<p align="center">
  <img src="https://github.com/digbangbang/Learning/assets/78746384/4a8b4e02-bbaa-4607-9c84-fea009bba0e8" alt="legend">
</p>

## [BERTEmbedder](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/encoders/modules.py#L80)

### Using [BERTTokenizer](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/encoders/modules.py#L87) to map the text into integer

#### Example:
```python
from transformers import BertTokenizerFast

# Create a BertTokenizerFast object and load the vocabulary of the pretrained BERT model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Input text
input_text = "Hugging Face Transformers library is awesome!"

# Step1：tokenize
tokens = tokenizer.tokenize(input_text)
# output: ['hugging', 'face', 'transformers', 'library', 'is', 'awesome', '!']

# Step2：add special sign
tokens = ["[CLS]"] + tokens + ["[SEP]"]
# output: ['[CLS]', 'hugging', 'face', 'transformers', 'library', 'is', 'awesome', '!', '[SEP]']

# Step3：encode
input_ids = tokenizer.convert_tokens_to_ids(tokens)
# output: [101, 17662, 2227, 17061, 4603, 2003, 12476, 999, 102]

# Step4：padding and truncation
# Suppose the model requires an input length of 10, we pad or truncate to meet this length
max_length = 10
input_ids = input_ids[:max_length] + [0] * (max_length - len(input_ids))
# output: [101, 17662, 2227, 17061, 4603, 2003, 12476, 999, 102, 0]

# Final
print("Original text:", input_text)
print("Processed tokens:", tokens)
print("Processed input_ids:", input_ids)
```

### Using [Transformer](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/encoders/modules.py#L89) to embedding the token(already map to the input_ids)

From BERTTokenizer, text's shape didn't change, keeping the (batch_size, num_sequence). 

Then going through the [TransformerWrapper](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/x_transformer.py#L548C7-L548C25), the output's shape will become (batch_size, num_sequence, dim).

## LDM

### Using [EMA](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py#L90) to update the parameters

$$ S_t = \beta \cdot S_{t-1} + (1 - \beta) \cdot X_t $$

In nn.Module, using `self.register_buffer()` to save parameters and update without backpropagation. And the saved $S_t$ will be called the shadow parameters. $X_t$ is the original parameters, which can be updated by backpropagation.

#### EMA advantage:
- Smooth the change trend of model parameters
- Improve the robustness of the model
- Preventing overfitting

### Using [Variational Lower Bound](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py#L314) to strengthen the loss

#### $L1$ or $L2$ [loss](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py#L307) in DDPM's UNet

This can also be called $L_{simple}$

$$\mathbb{E} _{\mathbf{x} _0, \boldsymbol{\epsilon}} \left \| \boldsymbol{\epsilon}-\boldsymbol{\epsilon} _\theta\left(\sqrt{\bar{\alpha} _t} \mathbf{x} _0+\sqrt{1-\bar{\alpha} _t} \boldsymbol{\epsilon}, t\right) \right \| \quad or \quad \mathbb{E} _{\mathbf{x} _0, \boldsymbol{\epsilon}} \left \| \left \| \boldsymbol{\epsilon}-\boldsymbol{\epsilon} _\theta\left(\sqrt{\bar{\alpha} _t} \mathbf{x} _0+\sqrt{1-\bar{\alpha} _t} \boldsymbol{\epsilon}, t\right) \right \|\right \|^2$$

#### Variational Lower Bound in DDPM

$$\mathbb{E} _{\mathbf{x} _0, \boldsymbol{\epsilon}}\left[\frac{\beta _t^2}{2 \sigma _t^2 \alpha _t\left(1-\bar{\alpha} _t\right)}\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon} _\theta\left(\sqrt{\bar{\alpha} _t} \mathbf{x} _0+\sqrt{1-\bar{\alpha} _t} \boldsymbol{\epsilon}, t\right)\right\|^2\right]$$

$L_{simple}$ and $L_{VLB}$ are different in the weight.

#### [Loss in DDPM](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py#L294)

```python
def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
```














