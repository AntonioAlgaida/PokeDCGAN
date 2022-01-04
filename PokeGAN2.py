#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 08:25:40 2022

@author: antonio
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid,save_image
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

#%%
from torchvision.transforms.transforms import RandomRotation
#Hyperparameter
batch_size = 64
image_size = 96
normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
data_dir = r'PokemonData'
normal_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(*normalization_stats)]))

# Augment the dataset with mirrored images
mirror_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(*normalization_stats)]))

# Augment the dataset with color changes
color_jitter_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
    transforms.ToTensor(),
    transforms.Normalize(*normalization_stats)]))

#%%
data_loader = DataLoader(dataset=ConcatDataset([normal_dataset,mirror_dataset,color_jitter_dataset]), batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=False)

#%%
def denorm(image):
    return image * normalization_stats[1][0] + normalization_stats[0][0]

#%%
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dataloader, nmax=64):
    for images, _ in dataloader:
        show_images(images, nmax)
        break

#%%
show_batch(data_loader)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#%%
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
#%%
discriminator = nn.Sequential(
    # Input is 3 x 64 x 64
    nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1, bias=False),
    nn.InstanceNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 64 x 32 x 32

    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
    nn.InstanceNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 16 x 16

    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
    nn.InstanceNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 8 x 8

    nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=1, bias=False),
    nn.InstanceNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 4 x 4
    
    # nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False),
    # nn.InstanceNorm2d(1024),
    # nn.LeakyReLU(0.2, inplace=True),
    
    # With a 4x4, we can condense the channels into a 1 x 1 x 1 to produce output
    nn.Conv2d(1024, 1, kernel_size=5, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)
discriminator.to(device)
discriminator.apply(weights_init)

#%%
seed_size = 100

generator = nn.Sequential(
    # Input seed_size x 1 x 1
    nn.ConvTranspose2d(seed_size, 1024, kernel_size=3, padding=0, stride=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    # Layer output: 256 x 4 x 4

    nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # Layer output: 128 x 8 x 8

    nn.ConvTranspose2d(512, 256, kernel_size=5, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # Layer output: 64 x 16 x 16

    nn.ConvTranspose2d(256, 128, kernel_size=5, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # Layer output: 32 x 32 x 32
    
    nn.ConvTranspose2d(128, 64, kernel_size=5, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # Layer output: 32 x 32 x 32
    
    nn.ConvTranspose2d(64, 3, kernel_size=6, padding=1, stride=2, bias=False),
    nn.Tanh()
    # Output: 3 x 64 x 64
)
generator.to(device)
generator.apply(weights_init)
#%%
def train_discriminator(real_pokemon, disc_optimizer):
    # Reset the gradients for the optimizer
    disc_optimizer.zero_grad()

    # Train on the real images
    real_predictions = discriminator(real_pokemon)
    # real_targets = torch.zeros(real_pokemon.size(0), 1, device=device) # All of these are real, so the target is 0.
    real_targets = torch.rand(real_pokemon.size(0), 1, device=device) * (0.1 - 0) + 0 # Add some noisy labels to make the discriminator think harder.
    real_loss = F.binary_cross_entropy(real_predictions, real_targets) # Can do binary loss function because it is a binary classifier
    real_score = torch.mean(real_predictions).item() # How well does the discriminator classify the real pokemon? (Higher score is better for the discriminator)

    # Make some latent tensors to seed the generator
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)

    # Get some fake pokemon
    fake_pokemon = generator(latent_batch)

    # Train on the generator's current efforts to trick the discriminator
    gen_predictions = discriminator(fake_pokemon)
    # gen_targets = torch.ones(fake_pokemon.size(0), 1, device=device)
    gen_targets = torch.rand(fake_pokemon.size(0), 1, device=device) * (1 - 0.9) + 0.9 # Add some noisy labels to make the discriminator think harder.
    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
    gen_score = torch.mean(gen_predictions).item() # How well did the discriminator classify the fake pokemon? (Lower score is better for the discriminator)

    # Update the discriminator weights
    total_loss = real_loss + gen_loss
    total_loss.backward()
    disc_optimizer.step()
    
    # torch.save(discriminator.state_dict(), 'ckpts/dis.pth')
    return total_loss.item(), real_score, gen_score
#%%
def train_generator(gen_optimizer):
    # Clear the generator gradients
    gen_optimizer.zero_grad()

    # Generate some fake pokemon
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)
    fake_pokemon = generator(latent_batch)

    # Test against the discriminator
    disc_predictions = discriminator(fake_pokemon)
    targets = torch.zeros(fake_pokemon.size(0), 1, device=device) # We want the discriminator to think these images are real.
    loss = F.binary_cross_entropy(disc_predictions, targets) # How well did the generator do? (How much did the discriminator believe the generator?)

    # Update the generator based on how well it fooled the discriminator
    loss.backward()
    gen_optimizer.step()
    # torch.save(generator.state_dict(), 'ckpts/gen.pth')

    # Return generator loss
    return loss.item()

#%%
import os
from torchvision.utils import save_image

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(index, latent_batch, show=True):
    # Generate fake pokemon
    fake_pokemon = generator(latent_batch)

    # Make the filename for the output
    fake_file = "result-image-{0:0=4d}.png".format(index)

    # Save the image
    save_image(denorm(fake_pokemon), os.path.join(RESULTS_DIR, fake_file), nrow=8)
    print("Result Saved!")

    if show:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_pokemon.cpu().detach(), nrow=8).permute(1, 2, 0))

#%%
from tqdm.notebook import tqdm
import torch.nn.functional as F

# Static generation seed batch
fixed_latent_batch = torch.randn(64, seed_size, 1, 1, device=device)

def train(epochs, learning_rate, start_idx=1):
    # Empty the GPU cache to save some memory
    torch.cuda.empty_cache()
    
    # Track losses and scores
    disc_losses = []
    disc_scores = []
    gen_losses = []
    gen_scores = []
    
    # Create the optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    
    # Run the loop
    for epoch in range(epochs):
        # Go through each image
        for real_img, _ in tqdm(data_loader):
            real_img = real_img.to(device)
            # Train the discriminator
            disc_loss, real_score, gen_score = train_discriminator(real_img, disc_optimizer)

            # Train the generator
            gen_loss = train_generator(gen_optimizer)
        
        # Collect results
        disc_losses.append(disc_loss)
        disc_scores.append(real_score)
        gen_losses.append(gen_loss)
        gen_scores.append(gen_score)
        
        # Print the losses and scores
        print("Epoch [{}/{}], gen_loss: {:.4f}, disc_loss: {:.4f}, real_score: {:.4f}, gen_score: {:.4f}".format(
            epoch+start_idx, epochs, gen_loss, disc_loss, real_score, gen_score))
        
        # Save the images and show the progress
        save_results(epoch + start_idx, fixed_latent_batch, show=False)
        torch.save(generator.state_dict(), f'ckpts/gen_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'ckpts/dis_{epoch}.pth')

    # Return stats
    return disc_losses, disc_scores, gen_losses, gen_scores

def restart_train(epoch):
    if epoch > 0:
        generator.load_state_dict(torch.load(f'ckpts/gen_{epoch}.pth'))
        discriminator.load_state_dict(torch.load(f'ckpts/dis_{epoch}.pth'))
#%%
if __name__ == '__main__':
    learning_rate = 0.0002
    epochs = 5000
    # restart_train(86)
    hist = train(epochs, learning_rate)
#%%
def print_example():
    batch_size = 32
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)
    # Get some fake pokemon
    fake_pokemon = denorm(generator(latent_batch))
    plt.imshow(fake_pokemon[0].cpu().detach().permute(1, 2, 0))

#%%


#%%