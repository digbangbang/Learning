import os
import random
from dataclasses import dataclass

from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from diffusers import UNet2DModel
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from datasets import load_dataset

from matplotlib import pyplot as plt

from add.diffusion import forward_diffusion_process, backward_diffusion_process
from add.loops import train_epoch, sample_images_cfg, sample_images
from add.utils import add_zero_class, seed_everything

from PIL import Image

from datasets import load_dataset
from torchvision import transforms

from torch.utils.data import DataLoader

@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    batch_size = 32
    num_epochs = 100
    gradient_accumulation_steps = 8
    student_learning_rate = 3e-5
    discriminator_learning_rate = 3e-4
    lr_warmup_steps = 500

    num_student_steps = 4
    num_teacher_steps = 100

    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    device = "cuda"
    random_state = 42


config = TrainingConfig()

seed_everything(config.random_state)

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im




dataset = load_dataset("cifar10", split='train')

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["img"]] # CIFAR10 key is img
    return {"images": images, "label": examples["label"]}


dataset.set_transform(transform)

train_dataloader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True
)


repo_id = "google/ddpm-cifar10-32"
# model = UNet2DModel.from_pretrained(repo_id)

T = UNet2DModel.from_pretrained(repo_id)
S = UNet2DModel.from_pretrained(repo_id)

D = shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
D.fc = nn.Linear(1024, 2)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# S_noise_scheduler = DDPMScheduler(
#     num_train_timesteps=4,
#     beta_schedule="linear",
#     beta_start=0.0001,
#     beta_end=1.0
# )
# S_noise_scheduler.set_timesteps(num_inference_steps=4)

S_optimizer = torch.optim.AdamW(S.parameters(), lr=config.student_learning_rate)
D_optimizer = torch.optim.AdamW(D.parameters(), lr=config.discriminator_learning_rate)

S_scheduler = get_cosine_schedule_with_warmup(
    optimizer=S_optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

D_scheduler = get_cosine_schedule_with_warmup(
    optimizer=D_optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)















