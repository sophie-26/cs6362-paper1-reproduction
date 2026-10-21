# CS 6362 Assignment 2 — Paper 1 Reproduction

PyTorch reproduction of Figure 1(a), (b), and (c) from *Understanding Deep Learning Requires Rethinking Generalization* (Zhang et al., ICLR 2017).

## Overview

This project investigates whether neural networks can memorize randomized training labels and inputs, and how label corruption affects convergence time and generalization.

The experiments reproduce:

- **Figure 1(a):** Training loss under true labels, random labels, shuffled pixels, random pixels, and Gaussian noise.
- **Figure 1(b):** Relative convergence time under different label corruption probabilities.
- **Figure 1(c):** Test classification error under different label corruption probabilities.

## Implementation

The experiments use PyTorch and CIFAR-10, with:

- 5,000 training images and 1,000 test images
- MLP and approximate Inception and AlexNet architectures
- Label corruption probabilities of 0, 0.5, and 1.0
- SGD with momentum
- A maximum of 30 training epochs in quick mode
- Convergence defined as training classification error at or below 0.5%

## How to Reproduce

1. Open `figure1_reproduction (1).ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Run the notebook cells in order.
4. The notebook downloads CIFAR-10, trains the models, and generates all three reproduced figures.

Required Python packages include `torch`, `torchvision`, `numpy`, and `matplotlib`.

Training may take substantial time depending on available hardware.

## Limitations

This is a computationally reduced reproduction rather than an exact replication of the original experiments. The dataset is smaller, the convolutional architectures are approximations, and the corruption grid is reduced.

Some convolutional experiments did not reach the convergence threshold within the training budget. These results are retained and discussed in the accompanying report rather than replaced with fabricated convergence times.

## Reference

Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017). *Understanding Deep Learning Requires Rethinking Generalization*. ICLR 2017.

https://openreview.net/forum?id=Sy8gdB9xx
