---
tags:
title: CURL
subTitle: CURL
abbrlink: 67269

---
---
tags:
title: CURL
subTitle: CURL
abbrlink: 52309

---
<!--more-->

# CURL: Neural Curve Layers for Global Image Enhancement

## 主要贡献

* 基于多颜色空间的曲线调整块 CURL   神经修饰块。通过神经网络学习一条离散的点，代表一条曲线。使用该曲线分别在  Lab  RGB 以及 HSV空间 对图像进行全局调整。
* 提出了多颜色空间损失函数  就是每个颜色空间调整完后的结果都有个损失函数约束
* 改进了U-NET的编解码结构，提出TED的backbone。降低参数量的同时提升了性能



## 主要内容

### 主要结构

![image-20210114110645797](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120017.png)

主要流程从上图可以看出  首先使用一个编解码结构对输入图像处理，作者使用的是自己改进的TED结构做backbone。backbone输出一个特征图记为 F（多通道），取特征图 F 的前三通道为编解码器输出的RGB图像，后面的通道为特征。接着的CURL模块就是对这个RGB图做**全局调整**。首先将前三通道RGB图转为Lab格式，然后和剩余的特征图一起输入第一个块，输出得到一个向量（代表曲线），并使用这个曲线对Lab空间下的图调整后转为RGB格式 输入下一部分曲线调整模块。以此类推。

### TED

![image-20210114111442957](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120021.png)

TED的核心结构和UNET类似，右下角所示结构。和UNET的差别是，首先取消了UNET的每一层的跳跃连接，只保留了最高层的连接，并且这个连接作者称为 `MSCA-skip` 如图中红色的连线。MSCA结构是左边大图所示，有三个分支，最上面的分支是全局连接，有几个步长为2的卷积加最后一个全连接。中间分支是卷积率为2的分支，最下面是卷积率为4的分支。最后concat 1x1卷积压缩输出。输入输出特征图尺寸通道数一致。作者对比了他提出的这个TED 结构和 UNET的准确率参数量，可以看出参数量较少但是准确率还比较高。同时还看出只有第一层跨层连接参数量少很多但是效果不输全部都连接的情况。

另外，作者还讨论了两种输入模式，RGB-to-RGB, RAW-to-RGB 两种情况。对于RAW格式的输出，需要稍微欸修改backbone的编码结构，具体的就是使用 pixel-shuffle层将输入RAW格式 转化为  (H/r) (W/r) r^2 然后再输入bacnkbone的下采样路径，这样输出的尺寸是RGB的四分之一，再使用pixel shuffle的上采样方法获得和RGB输如一样大的特征图。RGB格式的输入就不使用 pixel shuffle操作了。RAW需要pixel-shuffle主要是因为RAW格式数据的特殊存储格式吧。

### CURL模块

![image-20210114113514372](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120025.png)

 可以看出 前面 backbone输出的是一个 特征图 取特征图前3通道（蓝色部分为RGB图）曲线就是对这个进行全局调整的。神经网络的输出曲线是由全连接层产生，即离散的点。调整公式如下。 M代表输出点的个数，km代表第m个点的值，I 代表输入像素值，S代表输出的调整缩放因子。因此最终的调整结果为  S*I  **这样的另一个巧妙之处在于**，例如HSV通道，可以使用 hue 对  saturation 进行调整（色相至饱和度曲线），换句话说缩放因子用 通道hue 计算，但是用算出来的缩放因子对saturation 通道调整。

> we arrange the neural curve layers in a particular sequence, adjusting firstly luminance and the a, b chrominance channels (using three curves respectively) in CIELab space. Afterwards, we adjust the red, green, blue channels (using three curves respectively) in RGB space. Lastly hue is scaled based on hue, saturation based on saturation, **saturation based on hue**, and value based on value (using four curves respectively) in HSV space



![image-20210114113729720](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120028.png)

这个公式这么理解：

![image-20210114115936548](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120032.png)



### 损失函数

上面的模块中涉及到 颜色空间的变换 作者使用的是 pytorch 可微分的变换实现。

![image-20210114120611354](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120040.png)

**为什么 要用乘积？** S V乘积相同但是 相反表示不同深浅的颜色，但是损失依然为0？所以为啥要乘积相等？

![image-20210114154837650](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120042.png)

![image-20210114154911861](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120043.png)

![image-20210114154919143](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120046.png)

最后一个损失很明显是约束曲线 的斜率 不要有太大的突变 防止过拟合  即相邻点之间的直线的斜率差距不要太大。

## 实验

作者在三个数据集上验证了算法  

* Samsung S7    90张训练 10张测试 10张验证  包含RAW/RGB图像对
* MIT-Adobe5k-DPE   5000张图，有专家调整的参考图   2250对训练图   500张测试  从训练集随机选择了500个做验证
* MIT-Adobe5k-UPE

### 消融实验

### 和其他方法对比

![image-20210114160519085](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120100.png)

![image-20210114160656667](https://cdn.jsdelivr.net/gh/changruowang/cloudimg/img/20210424120104.png)

这是测试
