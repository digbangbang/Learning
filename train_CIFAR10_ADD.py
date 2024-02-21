
import os
import random
import torch
import numpy as np
import torchvision

from dataclasses import dataclass

from tqdm import tqdm

from torch import nn

from PIL import Image

from datasets import load_dataset

from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights





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

def seed_everything(seed: int,
                    use_deterministic_algos: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)


def forward_diffusion_process(x, noise_scheduler, device: str = "cuda", num_timesteps: int = 1000):
    bs = x.shape[0]
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)

    noise = torch.randn_like(x)
    random_t_index = random.randint(0, num_timesteps - 1)
    t = noise_scheduler.timesteps[random_t_index]
    t_batch = torch.full(
        size=(x.shape[0],),
        fill_value=t,
        dtype=torch.long
    ).to(device)

    noisy_x = noise_scheduler.add_noise(x, noise, t_batch)

    return noisy_x, random_t_index


def backward_diffusion_process(x, t, model, noise_scheduler, device: str = "cuda", num_timesteps: int = 1000):
    bs = x.shape[0]
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)

    for t in noise_scheduler.timesteps[-t - 1:]:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(bs,),
            fill_value=t.item(),
            dtype=torch.long
        ).to(device)

        noise_pred = model(
            model_input, t_batch, return_dict=False
        )[0]

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x

def train_epoch(S, T, D, S_optimizer, D_optimizer,t_noise_scheduler, s_noise_scheduler, accelerator, adversarial_loss, reconstruction_loss, lambd,
                dataloader, device: str = "cuda", num_student_timesteps: int = 4, num_teacher_timesteps: int = 1000):
    S.train()
    T.eval()
    D.train()

    S_losses = []
    D_losses = []


    for batch in tqdm(dataloader):
        x = batch["images"]

        noise = torch.randn_like(x).to(device)

        bs = x.shape[0]
        target_real = torch.ones((bs,), dtype=torch.long, device=device)
        target_fake = torch.zeros((bs,), dtype=torch.long, device=device)

        # Foward diffusion process on clean image
        xs, s = forward_diffusion_process(x, s_noise_scheduler, num_timesteps=num_student_timesteps)

        # Train ADD-student
        # Backward diffusion process on noised image, using noisy_images and t
        S_optimizer.zero_grad()
        x_theta = backward_diffusion_process(xs, s, S, s_noise_scheduler, num_timesteps=num_student_timesteps)
        L_G_adv = adversarial_loss(D(x_theta), target_real)

        # Forward diffusion process on an image denoised by ADD-student
        # xt, t = forward_diffusion_process(x, t_noise_scheduler, num_timesteps=num_teacher_timesteps) # original code
        xt, t = forward_diffusion_process(x_theta, t_noise_scheduler, num_timesteps=num_teacher_timesteps)
        with torch.no_grad():
            x_psi = backward_diffusion_process(xt, t, T, t_noise_scheduler, num_timesteps=num_teacher_timesteps)

        с = 1 / (t + 1)
        d = reconstruction_loss(x_theta, x_psi) * с  # * c(t), where c(t) = a_t

        S_loss = L_G_adv + lambd * d

        # accelerator.clip_grad_norm_(S.parameters(), 1.0)
        accelerator.backward(S_loss)
        S_optimizer.step()

        # Train Discriminator
        D_optimizer.zero_grad()
        real_loss = adversarial_loss(D(x), target_real)  # TODO: Need R1 regularization
        fake_loss = adversarial_loss(D(x_theta.detach()), target_fake)
        L_D_adv = (real_loss + fake_loss) / 2

        # accelerator.clip_grad_norm_(D.parameters(), 1.0)
        accelerator.backward(L_D_adv)
        D_optimizer.step()

        S_losses.append(S_loss.item())
        D_losses.append(L_D_adv.item())

    S_losses, D_losses = sum(S_losses) / len(dataloader.dataset), sum(D_losses) / len(dataloader.dataset)

    return S_losses, D_losses

def sample_images(model, noise_scheduler, device: str = "cuda", c: int = 0, bs: int = 16,
                  num_inference_steps: int = 1000):
    model.eval()
    model.to(device)

    x = torch.randn((bs, 3, 32, 32)).to(device)
    # x = torch.randn((bs, 3, 32, 32))

    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(x.shape[0],),
            fill_value=t.item(),
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            noise_pred = model(
                model_input,
                t_batch,
                return_dict=False
            )[0]

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x

@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    batch_size = 256
    num_epochs = 100
    gradient_accumulation_steps = 8
    student_learning_rate = 3e-5
    discriminator_learning_rate = 3e-4
    lr_warmup_steps = 200

    num_student_steps = 4
    num_teacher_steps = 100

    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    device = "cuda"
    random_state = 42


config = TrainingConfig()

seed_everything(config.random_state)


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

t_noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

s_noise_scheduler = DDPMScheduler(
    num_train_timesteps=4,
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=1.0
)
# s_noise_scheduler.set_timesteps(num_inference_steps=4)

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



accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)

train_dataloader, S, T, D, S_optimizer, D_optimizer = accelerator.prepare(
    train_dataloader, S, T, D, S_optimizer, D_optimizer
)


for epoch in range(100):
    S_loss, D_loss = train_epoch(S, T, D,
                S_optimizer, D_optimizer,
                t_noise_scheduler=t_noise_scheduler,
                s_noise_scheduler=s_noise_scheduler,
                accelerator=accelerator,
                adversarial_loss=nn.CrossEntropyLoss().to(config.device),
                reconstruction_loss=nn.MSELoss().to(config.device),
                lambd=1.0,
                dataloader=train_dataloader,
                device=config.device,
                num_student_timesteps=config.num_student_steps,
                num_teacher_timesteps=config.num_teacher_steps,
               )

    if (epoch + 1) % 5 == 0:
        generated_images = sample_images(S, s_noise_scheduler, config.device, num_inference_steps=config.num_student_steps)
        pil_images = show_images(generated_images)
        # plt.imshow(pil_images)
        pil_images.save(f"ADD_CIFAR10_result/{epoch}.jpg")

        torch.save(S, f"ADD_CIFAR10_result/S_{epoch}.pt")
        torch.save(D, f"ADD_CIFAR10_result/D_{epoch}.pt")











