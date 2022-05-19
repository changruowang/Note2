# 自监督图像去噪系列

2018 ICML2018的 N2N往后，有一些列的图像自监督去噪算法，不需要参考图像，仅仅通过噪声图像训练网络。

<!--more-->

系列文章

- [Noise2Noise](https://arxiv.org/pdf/1803.04189.pdf)
- [Probabilistic Noise2Void](https://arxiv.org/pdf/1906.00651.pdf)
- [Noise2Void_ICCV2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)
- [Noise2Self_ICML2019](http://proceedings.mlr.press/v97/batson19a/batson19a.pdf)
- [Self-Guided_ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Self-Guided_Network_for_Fast_Image_Denoising_ICCV_2019_paper.pdf)
- [Noisier2Noise_CVPR2020](https://arxiv.org/pdf/1910.11908.pdf)
- [Self2Self_CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)

[参考链接](https://blog.csdn.net/zbwgycm/article/details/81134631)

[参考链接](https://blog.csdn.net/weixin_36474809/article/details/86535639)

主要理解就是  图像去噪、超分等恢复任务，用CNN学习的其实是 参考图像的一个平均。例如LR图像由于边缘，细节信息丢失，它和参考HR图像是个一对多的分布。就是一个低质量图像可能会对应多个参考图像，直接用CNN的L2损失计算  成对损失，最终学出来的结果其实是 多个参考图像的平均。基于这个考虑，如果给成对的参考图像和输入图像同时加上 一个 **均值为0** 的随机噪声，那么学习的就是有噪声的参考图像的均值，而由于加上的是0均值的噪声 加上噪声后的均值 趋近于原始不加噪声的参考图像的均值。

