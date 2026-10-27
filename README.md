# Dynamic Preprocessing for Domain Generalization with Episodic Test-Time Adaptation in White Blood Cell Classification

This repository contains the PyTorch implementation associated with the paper *Dynamic preprocessing for domain generalization with episodic test-time adaptation in white blood cell classification*.

The proposed method replaces the initial convolutional layer of a classification network with an input-conditioned dynamic preprocessing module. A lightweight kernel generator with 9,110 trainable parameters produces 16 image-specific convolutional kernels of size 3 × 5 × 5.

The framework is evaluated using MobileNetV2 and ArConv on three white blood cell datasets: MLL23, Matek-19, and Raabin-WBC.

## Experimental Protocol

We use leave-one-domain-out evaluation. For each experiment, two datasets are used as source domains, and the remaining dataset is held out for testing. The main classification results are reported over five independent training runs.

The proposed episodic test-time adaptation (TTA) procedure generates confidence-selected pseudo-labels from unlabeled target mini-batches. It performs at most one adaptation update per mini-batch and restores the source-trained parameters before processing the next mini-batch.

Only the kernel generator and the designated trainable parameters of the initial convolutional stage are updated during TTA. The deeper classification backbone and classifier remain frozen.

## Reproducibility

The implementation includes model definitions and supporting components for dynamic preprocessing, domain-generalization evaluation, and episodic TTA. The paper and its Supplementary Material describe the dataset harmonization, training settings, evaluation metrics, and experimental results.
