# Awesome Video Prediction [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome video prediction papers with brief summary.

## Table of Contents
* [Blogs](#Blogs)
* [Surveys](#Surveys)
* [Papers](#Papers)

## Blogs
...

## Surveys
* ★ [A Review on Deep Learning Techniques for Video Prediction](https://arxiv.org/abs/2004.05214) | TPAMI 2020
* [Deep Learning for Vision-based Prediction: A Survey](https://arxiv.org/abs/2007.00095) | Arxiv 2020

## Papers
* **Baseline Video Language Modeling** (**BVLM**) | [Video (language) modeling: a baseline for generative models of natural videos](https://arxiv.org/abs/1412.6604) | Arxiv 2014 FAIR NYU
  * first video prediction | patch-level language model, recurrent CNN | no inductive bias, raw pixels

* **LSTM Encoder-Decoder** (**LSTM-ED**) | [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681) | ICML 2015
  * unsupervised learning representation | LSTM encoder into representation and LSTM decoder to reconstruct, FC-LSTM | no inductive bias, raw pixels

* **Convolutional LSTM** (**ConvLSTM**) | [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214) | NeurIPS 2015 HKUST
  * model well spatial correlations | just modified to convLSTM as LSTM-ED, convLSTM | no inductive bias, raw pixels

* **Predictive Generative Network** (**PGN**) | [Unsupervised learning of visual structure using predictive generative networks](https://arxiv.org/abs/1511.06380) | Arxiv 2015 Harvard
  * unsupervised learning representation | CNN-LSTM-deCNN and mse+adversarial loss, CNN+LSTM+GAN | no inductive bias, raw pixels

* **Predictive Coding Network** (**PredNet**) | [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104) | Arxiv 2016 Harvard
  * unsupervised learning representation | stacked multi-level encode representation and decode reconstruction variant, convLSTM | no inductive bias, raw pixels

* **Beyond Mean Square Error** (**BeyondMSE**) | [Deep multi-scale video prediction beyond mean square error]([[1511.05440\] Deep multi-scale video prediction beyond mean square error (arxiv.org)](https://arxiv.org/abs/1511.05440)) | ICLR 2016 FAIR NYU (Yann LeCun)
  * deal with blur | adversarial loss + gradient difference loss, CNN+GAN | no inductive bias, raw pixels

* **Predictive Recurrent Neural Network** (**PredRNN**) | [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504) | NeurIPS 2017 TPAMI 2022 Tsinghua (Yunbo Wang)
  * solve several problems in design of convLSTM for spatiotemporal predictive learning | spatiotemporal memory flow + spatiotemporal LSTM + reverse scheduled sampling curriculum learning, convLSTM | no inductive bias, raw pixels

* **Improved Predictive Recurrent Neural Network** (**PredRNN++**) | [PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning](https://arxiv.org/abs/1804.06300) | ICML 2018 Tsinghua (Yunbo Wang)
  * deeper in time and deep-in-time RNN vanishing gradient | causal LSTM + gradient highway unit, convLSTM | no inductive bias, raw pixels

* **Eidetic 3D LSTM** (**E3D-LSTM**) | [Eidetic 3D LSTM: A Model for Video Prediction and Beyond](https://openreview.net/forum?id=B1lKS2AqtX) | ICLR 2019 Tsinghua (Yunbo Wang, Fei-Fei Li)
  * learn good for both short-term and long-term | 3D CNN for local dynamics and recurrent modeling for temporal dependencies, 3D CNN+LSTM | no inductive bias, raw pixels

* ★ **Convolutional Dynamic Neural Advection** (**CNDA**) | [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157) | NeurIPS 2016 UCBerkeley (Chelsea Finn, Ian Goodfellow, Sergey Levine)
  * first real-world video long-range prediction | explicitly model pixel motion then merge previous frame, convLSTM | kernel-based transformation

* **Object-centric Transformation** (**ObjectTransformation**) | [Learning Object-Centric Transformation for Video Prediction](https://dl.acm.org/doi/10.1145/3123266.3123349) | ACM-MM 2017 PKU
  * different objects motion | attention to object patches and predict transformation kernels, CNN+RNN | kernel-based transformation

* **Spatially-Displaced Convolution Network** (**SDC-Net**) | [SDC-Net: Video prediction using spatially-displaced convolution](https://arxiv.org/abs/1811.00684) | ECCV 2018 Nvidia
  * high-resolution video prediction | combine vector-based and kernel-based transformation, 3D CNN | vector-based transformation + kernel-based transformation

* ★ **Motion-Content Network** (**MCnet**) | [Decomposing Motion and Content for Natural Video Sequence Prediction](https://arxiv.org/abs/1706.08033) | ICLR 2017
  * first decompose motion and content | motion encoder + content encoder + combination decoder, CNN+convLSTM | motion and content separation

* **Decompositional Disentangled Predictive Auto-Encoder** (**DDPAE**) | [Learning to Decompose and Disentangle Representations for Video Prediction](https://arxiv.org/abs/1806.04166) | NeurIPS 2018 Stanford (Li Fei-Fei)
  * deal with high-dimentionality | decompose whole frame to different components and disentangle each component to time-invariant content and low-dimensionality pose, CNN+RNN+VAE | vector-based transformation + motion and content separation

* ★ **Spatial-Temporal Multi-Frequency Analysis Network** (**STMANet**) | [Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction](https://arxiv.org/abs/2002.09905) | CVPR 2020 CAS
  * deal with image distortion and temporal inconsistency | merge multi-level both spatial and temporal wavelet analysis into prediction, CNN+LSTM+wavelet | add traditional CV, raw pixels

* ★ **Stochastic Variational Video Prediction** (**SV2P**) | [Stochastic Variational Video Prediction](https://arxiv.org/abs/1710.11252) | ICLR 2018 UIUC (Chelsea Finn, Sergey Levine)
  * first introduce stochastic | VAE noise as stochastic condition for CDNA, 3D CNN+convLSTM+VAE | kernel-based transformation + VAE stochastic

* **Stochastic Video Generation with a Learned Prior** (**SVG-LP**) | [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) | ICML 2018 NYU
  * "learned prior as uncertainty predictive model" | learned prior for VAE, convLSTM+VAE | VAE stochastic

* **Stochastic Adversarial Video Prediction** (**SAVP**) | [Stochastic Adversarial Video Prediction](https://arxiv.org/abs/1804.01523) | ICLR 2019 UCBerkeley (Chelsea Finn, Sergey Levine)
  * bring together stochastic and realistic | VAE-GAN for SV2P, 3D CNN+convLSTM+VAE+GAN | kernel-based transformation + VAE stochastic

* **Hierarchical VRNN** (**Hierarchical-VRNN**) | [Improved Conditional VRNNs for Video Prediction](https://arxiv.org/abs/1904.12165) | ICCV 2019
  * "still blurry and due to underfitting" | hierarchical levels of latents to increase expressiveness, CNN+RNN+VAE | VAE hierarchical stochastic

* **Greedy Hierarchical Variational Auto-Encoders** (**GHVAE**) | [Greedy Hierarchical Variational Autoencoders for Large-Scale Video Prediction](https://arxiv.org/abs/2103.04174) | CVPR 2021 Stanford (Li Fei-Fei, Chelsea Finn)
  * deal with memory constraints and optimization instability problems for hierarchical VAE | greedy and modular optimization, CNN+RNN+VAE | VAE hierarchical stochastic

* **Video Diffusion Models** (**VDM**) | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | Arxiv 2022 Google (Jonathan Ho)
  * first video diffusion model for primarily unconditional video generation | diffusion model with 3D U-Net, 3D CNN+diffusion | no inductive bias, raw pixels

* **Masked Conditional Video Diffusion** (**MCVD**) | [MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853) | NeurIPS 2022
  * general-purpose as prediction/generation/interpolation | conditioned on masked past or future frames U-Net, CNN+diffusion | no inductive bias, raw pixels

* **Residual Video Diffusion** (**RVD**) | [Diffusion Probabilistic Modeling for Video Generation](https://arxiv.org/abs/2203.09481) | Arxiv 2022
  * "residual errors are easier to model than future observations" | MAF for average + diffusion for residual, CNN+RNN+diffusion | no inductive bias, raw pixels

* **Flexible Diffusion Model** (**FDM**) | [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495) | Arxiv 2022
  * deal with long duration coherent prediction | randomly sampling train, 3D CNN+diffusion | no inductive bias, raw pixels
