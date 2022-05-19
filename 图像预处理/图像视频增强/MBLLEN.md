---
tags:
title: MBLLEN
subTitle: MBLLEN
abbrlink: 61042

---
---
title: MBLLEN
abbrlink: 12562
mathjax: true
abstract:
tags:
	- 低光照图像增强
password:
---

<!--more-->

# MBLLEN

## 主要思想

​		由于图像内容的复杂性，作者认为简单的网络难以实现高质量的图像增强。因此设计了MBLLEN 的多分枝结构。将图像增强任务分解成和不同特征相关的子问题，不同特征层分别增强，最后通过多分枝结果融合得到搞质量的输出。



## 网络结构

### 结构

​		由三部分组成，FEM, EM, FM;特征提取模块，增强模块，融合模块。

1. FEM：将三通道微光图像输入FEM模块，FEM模块实际是由10个步长1，3X3卷积Relu层组成的网络。每一层卷积后的feature map一方面作为EM模块的输入，一方面接着传给下一层卷积接着提特征
2. EM：增强模块，数量等于上一层输出的feature map的个数。每个EM模块都是conv deconv结构，输出的尺寸和原微光图像尺寸相同。所有EM模块输出的featuremap contact 作为FM的输出
3. FM：多分支融合，将上一层contact的结果用1X1卷积聚合。得到最终的输出
4. 应用在视频....

![1572958613424](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212135.png)

### 损失函数

​		传统的MSE或者MAE损失在增强的任务中不能很好的表现，作者使用了更复杂的损失函数，包含结构损失，Context(语义)损失， 区域损失

1. 结构损失（structure loss ）：这个损失是为了提高输出图像的视觉 效果。通常低光照图像暗区因为硬件捕获的问题带有模糊和伪影，他们视觉效果不好但是不能被MAE损失表现出来。作者提出的结构损失中包含两部分：简化的SSIM MS-SSIM。

$$
L_{S S M}=-\frac{1}{N} \sum_{p \in i m g} \frac{2 \mu_{x} \mu_{y}+C_{1}}{\mu_{x}^{2}+\mu_{y}^{2}+C_{1}} \cdot \frac{2 \sigma_{x y}+C_{2}}{\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}}
$$

$$
L_{S t r}=L_{S S I M}+L_{M S-S S I M}
$$

2. context loss： 作者认为MSE和SSIM智能表示低层信息，认为使用更高级的语义信息是有必要的。采用了SRGAN中相似的做法来设计损失。具体是使用VGG-19 net提取两张图片的特征图。然后比较特征图的差别，如下式，i,j表示VGG-19中第j层特征第i个block的输出特征图。

$$
L_{V G G / i, j}=\frac{1}{W_{i, j} H_{i, j} C_{i, j}} \sum_{x=1}^{W_{i, j}} \sum_{z=1}^{C_{i, j}}\left\|\phi_{i, j}(E)_{x, y, z}-\phi_{i, j}(G)_{x, y, z}\right\|
$$

3. **region loss**  区域损失：上面两个损失函数都是基于全图的。然而在图像增强任务中，需要对低光照区域提供更多的注意力。因此作者提出这个损失函数来平衡低光照区域和其他区域的损失。作者筛选暗区域的策略是 发现选取一副图中前40%暗的像素作为暗区域最能代表实际的暗区域。**这里可以寻找更恰当的选取暗区域的方法**  式子中，EL GL中的L代表输入图像的暗区域，H代表亮区域  E,G代表输出图像和groundtruth    WL=4  WH=1
   $$
   L_{R e g i o n}=w_{L} \cdot \frac{1}{m_{L} n_{L}} \sum_{i=1}^{n_{L}} \sum_{j=1}^{m_{L}}\left(\left\|E_{L}(i, j)-G_{L}(i, j)\right\|\right)+w_{H} \cdot \frac{1}{m_{H} n_{H}} \sum_{i=1}^{n_{H}} \sum_{j=1}^{m_{H}}\left(\left\|E_{H}(i, j)-G_{H}(i, j)\right\|\right)
   $$

### 实现细节

​		选取PASCAL VOC上的一部分图像使用损及gamma矫正转化为合成的低光照图像，同时加入了泊松噪声。56张验证，144张测试。minibatch 24  256X256X3的图像。VGG损失作者选取的是j=4 i=3处的feature map 。ADAM优化器，学习率0.002 b1=0.9 b2=0.99 e=10-8  学习率每个epoch learnrate * 0.95。

## 结果

![1572960634271](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212143.png)



## 结论

* 感觉这篇文章 出发点 很独特，不同于以往的基于retinex模型的方法，提取光照图等等。通过多层特征提取，然后分别增强，最后多分枝融合。思想很独特。
* 区域损失 值得借鉴和改进     VGG损失？？？ 



