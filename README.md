# StyleGAN.pytorch

This is pytorch StyleGAN implementation with smooth latent space interpolation generator, early stopping and optimized for RTX-2070 super GPU training


This repository contains the unofficial PyTorch implementation of the following paper:

> A Style-Based Generator Architecture for Generative Adversarial Networks <br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA) <br>
> http://stylegan.xyz/paper
> 
> Abstract: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.


## Features

- [x] Progressive Growing Training
- [x] Exponential Moving Average
- [x] Equalized Learning Rate
- [x] PixelNorm Layer
- [x] Minibatch Standard Deviation Layer
- [x] Style Mixing Regularization
- [x] Truncation Trick   
- [x] Using official tensorflow pretrained weights 
- [x] Gradient Clipping
- [ ] Multi-GPU Training
- [ ] FP-16 Support
- [ ] Conditional GAN

## How to use

### Requirements
- yacs
- tqdm
- numpy
- torch
- torchvision
- tensorflow(Optional, for ./convert.py)

### Running the training script:
Train from scratch:
```shell script
python train.py --config configs/sample.yaml
```

### Using trained model:
Resume training from a checkpoint (start form 128x128):
```shell script
python train.py --config config/sample.yaml --start_depth 5 --generator_file [] [--gen_shadow_file] --discriminator_file [] --gen_optim_file [] --dis_optim_file []
```
### Generate smooth interpolations

```shell script
python generate_samples.py --config config/sample.yaml --generator_file [] 
```

### Style Mixing

```shell script
python generate_mixing_figure.py --config config/sample.yaml --generator_file [] 
```

### Truncation trick

```shell script
python generate_truncation_figure.py --config configs/sample_cari2_128_truncation.yaml --generator_file cari2_128_truncation_gen.pth
```


## Reference

- **stylegan[official]**: https://github.com/NVlabs/stylegan
- **pro_gan_pytorch**: https://github.com/akanimax/pro_gan_pytorch
- **pytorch_style_gan**: https://github.com/lernapparat/lernapparat

## Thanks

Please feel free to open PRs / issues / suggestions here.

## Due Credit
This code heavily uses NVIDIA's original 
[StyleGAN](https://github.com/NVlabs/stylegan) code. We accredit and acknowledge their work here. The 
[Original License](/LICENSE_ORIGINAL.txt) 
is located in the base directory (file named `LICENSE_ORIGINAL.txt`).
