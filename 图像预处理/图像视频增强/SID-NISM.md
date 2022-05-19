---
title: SID-NISM
abbrlink: 37892
abstract:
tags:
	- 低光照图像增强
password:
---


<!--more-->

# SID-NISM: A Self-supervised Low-light Image Enhancement Framework

## 主要思路

1. 类似RetinexNet 的方法构建分解网络，但是在是无监督的 多以区别在于  它 将低光照图的  直方图均衡化版本作为高亮度版本一起分解，构建场景一致性损失。
2. 考虑了噪声，I = R X L + N
3. 第二阶段的 光照调整方法 不是gamma 矫正，是作者自己提出的新的函数  这个可以借鉴



## 主要内容

![image-20201231103107211](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212012.png)

### 损失函数

![image-20201231103305198](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212017.png)

均衡后的图像的 分解产生的 R 要和 低光照图直接分解产生的R 一致

![image-20201231103339686](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212022.png)

光照平滑和一致损失，第一项表明 光照分段平滑，以反射率图的梯度加权。第二项以高低光照图自身梯度加权，是为了让在两张图中都是边缘的地方损失小，非共同边缘的地方损失大，即两张光照图的边缘一致。这和RetinexNet中的损失一致

![image-20201231103645591](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212027.png)

第一项，意思为增强后的图像的 梯度要比原图像放大beta倍，当然 计算原输入图像 S 的梯度的时，滤掉了梯度较小的地方。第二项是，输入图像的 HSV中的H通道和 分解R的  H通道要一致，防止颜色乱变。

![image-20201231103920053](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212032.png)

最后一项是噪声 一致的 ，输入图像乘 噪声估计图N ，约束噪声的大小。（不知道为啥）

### 光照调整

![image-20201231104256103](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212037.png)

​		这个阶段是整篇文章比较有意思的地方。它分析了直接用GAMMA矫正来调整光照图带来的问题，然后针对它进行了改进，提出了一个新的光照调整函数。

​		具体的，图像R的对比度由于暗区域的过度增亮而被破坏。最终增强图像的照明水平仍然不足，因为明亮区域几乎没有变化。一句话总结就是，gamma矫正过分的提高了暗区域的光照，导致R*L 乘回去之后，R衰减的太少，使得整体效果显得过增强。对于亮区域gamma 矫正又几乎不调整，导致亮区域的低光光照被拉低。按它的思路应该是  暗区域亮度拉升变缓，以抑制R的过曝，亮区域亮度要再提高免得亮区域亮度乘上系数后又被压缩回去了。（是这样嘛？？？）

​		按照上述逻辑 作者提出的曲线波形是NIMS的形状，表达式为：

![image-20201231105830496](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212042.png)

![image-20201231105852294](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210508212047.png)

首先对光照图的像素亮度进行聚类，（两类）。分为亮像素和暗像素，去亮像素区的最小亮度值作为T，计算yita。参数 yita 的意义在于，在NISM下将亮像素的最小照明值映射到0.8 



