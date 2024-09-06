## What is GAN?

A generative adversarial network is a network that uses two networks called a generator and discriminator to generate synthetic data can convincingly mimic real data. For example for generating photorealistic images of people.

### generator

Learns to generate plausible data. The generated instances become negative training examples for the discriminator. During training, In its basic form GAN takes random noise as its input.

### discriminator

This is a classifier that analyzes data provided by the generator, and tries to identify if its is fake generated data or real data. It learns to distinguish the generator's fake data from the real data. The discriminator _penalizes the generator for producing fake results_.

During training, the discriminator connects to the two loss functions (The discriminator loss, and the generator loss). During training discriminator ignores the generator loss and just uses the discriminator loss. We use the generator loss during generator training.
During training:

1. It classifies both real data and fake data from the generator.
2. The discriminator loss penalizes the discriminator for misclassifying a real instance as fake or vice-versa.
3. The discriminator updates its weight during back-propagation.

## NOTE

GAN's convergence is hard to identify. That's its headache

## Resource

- [deeplearning ai](<https://github.com/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb>)
- [lightning bolts](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#basic-gan)
