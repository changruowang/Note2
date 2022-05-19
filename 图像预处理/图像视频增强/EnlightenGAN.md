---
tags:
title: EnlightenGAN
subTitle: EnlightenGAN
abbrlink: 68583

---
---
title: EnlightenGAN
abbrlink: 9266
mathjax: true
abstract:
	- 低光照图像增强
tags:
password:
---


<!--more-->

想解决的问题：低光照图像增强

方法：

- 提出非监督的生成对抗网络，EnlightenGAN，训练不需要低光度/正常光度图像对。
- 提出使用从输入本身提取的信息来规范化不成对的训练
- 一些创新：
  - global-local判断器结构
  - self-regularized perceptual loss fusion
  - 注意力机制

## Introduction

DL方法通常用成对图像作为输入，**缺点**：

1. 很难在实际中同时采集到同一场景的图像
2. 合成的图像不够真实，引入了许多人为因素
3. 对于低光照问题，没有唯一的或者说定义好的GT

解决的问题：

- 对低光照图像进行增强，但不需要成对的训练数据。

idea：

- 使用GAN在low和normal light之间建立unpaired mapping，但不依赖任何成对的图像
- 没有使用cycle-consistency作为prior工作
  - （不懂这个cycle-consistency是什么）
  - ref：[参考资料](https://zhuanlan.zhihu.com/p/70592331)
  - 一些point：
    - 最早是在朱俊彦的cycleGAN提出，用于在没有配对数据的情况下实现2个domain的image-to-image翻译。基本的思想，假设从X翻译到Y得到F(x)，再翻译回去G(F(x))，G(F(x))应该与X一模一样。
    - 
- 一些创新性工作
  - 1.提出dual-discriminator平衡全局和局部低光照增强x
  - 2.提出self-regularized perceptual loss来约束输入和增强版本后的特征距离
  - 3.提出开发的输入的光照信息作为self-regularized attentional map，在每个level的深度特征上regularize无监督学习

EnlightenGAN的创新点：

1. 第一个将unpaired训练引入低光照增强
2. self-regularization，通过保留自特征loss和自规则注意力机制实现
3. 增强来自不同domain的真实世界的低光照图像更加简单和灵活

## Related Works

### **Paired Datasets：Status Quo**

缺点：

- 数据量小，简单地增加或减少曝光时间会增加或减少局部的曝光。
- 在HDR领域，一些工作首先获取不同光照条件下的图像，然后对其进行排列和融合，但他们不是为单一图像后处理而设计的

### Traditional Approaches

经典方法：

- adaptive histogram equalization（AHE），自适应直方图平衡
  - Stephen M Pizer, E Philip Amburn, John D Austin, Robert Cromartie, Ari Geselowitz, Trey Greer, Bart ter Haar Romeny, John B Zimmerman, and Karel Zuiderveld. Adaptive histogram equalization and its variations. Com- puter vision, graphics, and image processing, 39(3):355– 368, **1987.**
- Retinex
  - Edwin H Land. The retinex theory of color vision. Scientific american, 237(6):108–129, **1977**
- multi-scale Retinex model
  - Daniel J Jobson, Zia-ur Rahman, and Glenn A Woodell. A multiscale retinex for bridging the gap between color images and the human observation of scenes. IEEE Transactions on Image processing, 6(7):965–976, **1997**
- 提出针对不均匀光照用bi-log信息平衡细节与自然感的增强算法
  - Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Nat- uralness preserved enhancement algorithm for non-uniform illumination images. IEEE Transactions on Image Process- ing, 22(9):3538–3548, **2013**
- 提出加权变分模型，估计reflectance和illumination
  - Xueyang Fu, Delu Zeng, Yue Huang, Xiao-Ping Zhang, and Xinghao Ding. A weighted variational model for simultane- ous reflectance and illumination estimation. In CVPR, pages 2782–2790, **2016**
- LIME，low-light image enhancement，先找到RGB中的最大值作为最初的光照估计，使用结构先验构建光照图
  - Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light im- age enhancement via illumination map estimation. IEEE Transactions on Image Processing, 26(2):982–993, **2017**
- 通过分解连续图像序列来同时处理低光照和去噪
  - Xutong Ren, Mading Li, Wen-Huang Cheng, and Jiaying Liu. Joint enhancement and denoising method via sequen-tial decomposition. In Circuits and Systems (ISCAS), 2018 IEEE International Symposium on, pages 1–5. IEEE, **2018**
- 提出更加robust的Retinex模型，与传统的Retinex模型对比，考虑了noise map，通过强噪声提高性能
  - Mading Li, Jiaying Liu, Wenhan Yang, Xiaoyan Sun, and Zongming Guo. Structure-revealing low-light image en- hancement via robust retinex model. IEEE Transactions on Image Processing, 27(6):2828–2841, **2018.**

### Deep Learning Approaches

目前大部分基于DL的方法都依赖于paired image，且图像大部分是从正常图像中人工合成的。

- LL-Net，堆叠的自动编码器，在patch level同时学习去噪和低光照增强。
  - Kin Gwn Lore, Adedotun Akintayo, and Soumik Sarkar. Ll- net: A deep autoencoder approach to natural low-light image enhancement. Pattern Recognition, 61:650–662, **2017**.
- Retinex-Net，设计了end-to-end框架，结合了Retinex理论
  - Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560, **2018**.
- HDR-Net ，将深度网络与双边网格处理（bilateral gird processing）、局部颜色仿射变换（local affine color transforms）
  - Micha¨el Gharbi, Jiawen Chen, Jonathan T Barron, SamuelW Hasinoff, and Fr´edo Durand. Deep bilateral learning for real- time image enhancement. ACM Transactions on Graphics (TOG), 36(4):118, **2017**
- 以及一些针对HDR领域的多帧低光照增强方法
  - Nima Khademi Kalantari and Ravi Ramamoorthi. Deep high dynamic range imaging of dynamic scenes. ACM Trans. Graph, 36(4):144, **2017**.
  - ShangzheWu, Jiarui Xu, Yu-Wing Tai, and Chi-Keung Tang. Deep high dynamic range imaging with large foreground motions. In Proceedings of the European Conference on Computer Vision (ECCV), pages 117–132, **2018**.
  - Jianrui Cai, Shuhang Gu, and Lei Zhang. Learning a deep single image contrast enhancer from multi-exposure images. IEEE Transactions on Image Processing, 27(4):2049–2062, **2018**.
- learning to see in the dark，直接在raw数据上，更注重避开放大的artifacts

### Adversarial Learning

使用GAN的方法同样使用的是paired训练数据。

一些人提出了无监督的GAN方法，使用对抗学习学习inter-domain。

提出两个方法，通过使用cycle-consistent loss+uppaired data，对两个不同领域之间进行翻译

- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle- consistent adversarial networks. In ICCV, pages 2223–2232, **2017**.
- Ming-Yu Liu, Thomas Breuel, and Jan Kautz. Unsupervised image-to-image translation networks. In Advances in Neural Information Processing Systems, pages 700–708, **2017**.

一些最新的工作基于上面的方法论应用在其他low level task（比如，去雾，去噪，SR，手机照片增强等）上：

- Xitong Yang, Zheng Xu, and Jiebo Luo. Towards percep tual image dehazing by physics-based disentanglement and adversarial training. In The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), **2018**.
- Yuan Yuan, Siyuan Liu, Jiawei Zhang, Yongbing Zhang, Chao Dong, and Liang Lin. Unsupervised image super- resolution using cycle-in-cycle generative adversarial net- works. CVPR Workshops, 30:32, **2018**.
- Yu-Sheng Chen, Yu-Ching Wang, Man-Hsin Kao, and Yung- Yu Chuang. Deep photo enhancer: Unpaired learning for image enhancement from photographs with gans. In Pro- ceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6306–6314, **2018**.
- Xin Jin, Zhibo Chen, Jianxin Lin, Zhikai Chen, and Wei Zhou. Unsupervised single image deraining with self- supervised constraints. arXiv preprint arXiv:1811.08575, **2018**.

EnlightenGAN采用unpaired训练，但是一个轻量级的one-pathGAN结构（即，没有cycle-consistency）,这样的好处是，训练稳定且简单。

## Method

网络结构：

[![img](https://github.com/GlassyWu/Note/raw/master/Paper/%E4%BD%8E%E5%85%89%E7%85%A7%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/img/EnlightenGAN.png)](https://github.com/GlassyWu/Note/blob/master/Paper/低光照图像增强/img/EnlightenGAN.png)

- 采用一个attention-guided U-Net作为生成器
- 用对偶的判断器判断全局和局部信息（即，一个global和一个local判断器）
- 使用一个自特征保留loss来引导训练，以及保留纹理和结构

#### （1）Global-Local Discriminators

用对抗loss来最小化真实和输出的正常光线的分布之间的距离。

image-level的判断器经常在空间变化的光照图像不work，在一些局部区域需要增强的部分和其他区域不太一样（比如，全局大部分是暗的，一个小部分是亮的，这种情况全局生成器可能就不能满足这个需求）。

解决方法：

作者设计了一个global-local判断器结构，都是用PatchGAN来判断真假。

- local判断器：

  - 在输出和正常光线图像随机裁剪局部patch，判断器学习判断这些patch的真假。

- global判断器：

  - 使用了相对判断器结构
    - relativistic discriminator structure：
      - Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018
  - 在标准的相对判断器上将sigmoid函数改为最小二乘法loss

  标准相对判断器公式： $$ D_{Ra}(x_{r},x_{f})=σ(C(x_{r})-E_{x_{f}∼P_{fake}}[C(x_{f})]) \ D_{Ra}(x_{f},x_{r})=σ(C(x_{f})-E_{x_{f}∼P_{real}}[C(x_{r})]) $$ 作者进行修改后的loss：

  **Global：** $$ L_{D}^{Global}=E_{X_{r}∼P_{real}}[(D_{Ra}(x_{r},x_{f})-1)^2]+E_{x_{f}∼P_{fake}}[D_{Ra}(x_f, x_r)^2] \ L_{G}^{Global}=E_{X_{f}∼P_{fake}}[(D_{Ra}(x_{f},x_{r})-1)^2]+E_{x_{r}∼P_{real}}[D_{Ra}(x_r, x_f)^2] $$ **Local:** $$ L_{D}^{Local}=E_{x_{r}∼P_{real-patches}}[(D(x_{r})-1)^2]+E_{x_{f}∼P_{fake-patches}}[(D(x_{f})-0)^2] \ L_{G}^{Local}=E_{x_{r}∼P_{fake-patches}[(D(x_{f})-1)^2]} $$

#### （2）Self Feature Preserving Loss

perceptual loss常用来限制提取的特征与GT尽可能接近，perceptual loss是利用预训练的VGG去模拟图像之间的特征空间距离。

**paper**：Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, pages 694–711. Springer, **2016**.

作者提出限制输入和输出的VGG-frature距离。

作者根据经验观察到当调整输入像素的密度范围，VGG的分类结果不是特别灵敏。（也被论文证实了）

设计了自正则化loss（self-regularzation loss)，以保留图像内容特征。

L_SFP的定义：

[![img](https://github.com/GlassyWu/Note/raw/master/Paper/%E4%BD%8E%E5%85%89%E7%85%A7%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/img/L_SFP.png)](https://github.com/GlassyWu/Note/blob/master/Paper/低光照图像增强/img/L_SFP.png)

对于裁剪的局部块也进行规则，用L_SFP_Local，此外，还在VGG特征图之后增加了instance normalization，目的是稳定训练。

#### （3）U-Net Generator Guided with Self-Regularized Attention

- U-Net作为backbone
- 将输入的RGB正则化到[0,1]，用1-I作为self-regularized attention map
- 然后resize attention map来适配每个feature map，然后与所有中间feature map、输出进行相乘
- 作者强调attention map也是self-regularization的一部分

attention-guided U-Net generator结构：

- 8个conv块 ，每个conv包括LeakyRelU+BN+2个3x3 conv
- 在上采样阶段，将标准deconvolutional layer替换为双线上采样层（bilinear upsampling）+一个卷积层，目的是减少checkerboard artifacts
  - (不懂bilinear upsampling，checkerboard artifacts)

## GAN相关论文：

1、Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle- consistent adversarial networks. In ICCV, pages 2223–2232, 2017.

2、Ming-Yu Liu, Thomas Breuel, and Jan Kautz. Unsupervised image-to-image translation networks. In Advances in Neural Information Processing Systems, pages 700–708, 2017

3、Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018.

## 其他

- HDR(high-dynamic-ranging)
  - 高动态范围成像，用来实现比普通数字图像技术更大曝光动态范围（即，更大的明暗差别）的一组技术
  - 目的：正确地表示真实世界中从太阳光直射到最暗的阴影这样大的范围亮度
- ablation analysis
  - 消融实验：分析不同参数或结构对实验结果产生的影响从而得到不同成分的作用
