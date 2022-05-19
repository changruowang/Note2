---
tags:
title: python语法
subTitle: python语法
abbrlink: 41152

---
---
title: python和pytorch常用语法
comments: true
tags:
  - python
abbrlink: 8787
---

<!--more--> 

# pytorch&python

## python *  ** 的用法

![1566285511946](figs/1566285511946.png)

### 用在函数的输入参数

​		在定义一个方法时，使用*表示输入参数列表不确定。会将输入的参数放入一个元组中，函数内可通过访问元组的方法访问里面的数据。使用**arg也可以表示不确定参数列表，但是会将输入参数打包为字典结构。因此函数内部可以通过访问字典结构的方法访问输入参数

```py
def myprint(*arg)
	print arg
myprint(1,2,3,4)
#打印的结果是一个元组

def myprint(*arg)
	for i in range(len(arg))
#输出1，2，3，4
```

```py
def myprint(**arg)
	print(arg)
myprint(a=1,b=2,c=3)
#打印结果：{'a':1, 'b':2, 'c':3}
```

### 解包

​		与上述过程相反

```py
def add（x，y）:
    print x+y
para =  (1,2)
add(*para)

def add(x,y):
    print x+y
kkwd = {'x' :1,'y':2}
add(**kkwd)
```



## 值传递，引用传递 ，可变对象，不可变对象

一般来说，将数字、字符串、元组是不可变对象，传入函数，相当于对内存进行了拷贝，在函数内修改其值不会改变函数外的值。但是对于列表、字典等可以增删的数据结构，传入函数后，相当于传递进去的是c语言中的指向地址的指针。所以在函数内对其进行有关地址的操作都可以修改函数外的变量的值。

参考链接：https://blog.csdn.net/qq_41987033/article/details/81675514



## 使用tensorboard数据可视化

### 安装

​		对于pytorch1.0 torch.utils.tensorboard中有这个文件，但是直接在程序中导入会报错，不考虑这个，自行安装tensorboardX。 需要安装tensorboardX为python代码执行，实现pytorch到tensorflow的转化生成一个日志文件，还需要安装tensorflow，才能调用tensorboard命令解析日志文件并上传到浏览器

 ```py
pip install tensorboardX
pip install tensorflow #安装的gpu版本
 ```

### 简单使用

​		在程序调用一下api确保生成日志文件

```pyth
writer = SummaryWriter('runs/fashion_mnist_experiment_1')  
writer.add_image('four_fashion_mnist_images', img_grid)
writer.close()   #必须有close 否则内容写不进去
```

​		在系统命令行输入以下命令

```python
tensorboard --logdir runs  
```

​		上述的runs为刚才在py中生成日志文件的路径，得确保该路径下存放得有生成得日志文件
​		输入上述命令后，会产生如下输出，在google浏览器打开http://DESKTOP-F3BBIIO:6006 如果浏览被拒，改为 http://localhost:6006。如果是本机电脑链接远程服务器，则将服务器名字改为服务器的ip地址



## pytorch的用法

### pytorch踩坑

* load模型时，他会自动将其加载在这个模型训练时所在的gpu上  此时若这个gpu刚好满了就不能加载。所以最好每次加自增加 map_location='cpu'  参数  load完毕后再to到需要的gpu上
* 

### cat stack unqueeze用法总结

​	相同： cat, stack 输入都为 1. tensor列表，2.另一个参数是维度 （n为输入tensor的维度数）

* cat : 将输入的tensor按照指定维度连接， 不要求输入的尺寸完全相同。dim的范围为 0~n-1  按照b, c, h, w的顺序对应dim从小到大。例如输入为二维矩阵，dim=0/1 =0表示在 h 维度拼接，即输出矩阵的行数为输入之和。
* stack: 将输入tensor增加维度堆叠，要求输入的尺寸完全相同。dim范围0~n 也是按照bchw的顺序对应dim。输出的tensor dim=n+1。对于输二维矩阵，dim=0，在c维度堆叠；dim=1，提取每个输入的行向量按行堆叠构成输出的一个通道，输入矩阵有多少行就有多少通道；dim=2，提取每个输入的行向量按列堆叠构成输出的一个通道，输入矩阵有多少行就有多少通道。对于输入三维张量，同理。
