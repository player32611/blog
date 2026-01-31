# 卷积神经网络

> **卷积神经网络**（Convolutional Neural Network，CNN）被用于图像识别、语音识别等各种场合，在图像识别的比赛中，基于深度学习的方法几乎都以 CNN 为基础。

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## 整体结构

CNN 和之前介绍的神经网络一样，可以像乐高积木一样通过组装层来构建。不过，CNN 中新出现了**卷积层**（Convolution 层）和**池化层**（Pooling 层）。

之前介绍的神经网络中，相邻层的所有神经元之间都有连接，这称为**全连接**（fully-connected），并用 Affine 层实现了全连接层。

![基于全连接层（Affine 层）的网络的例子](/images/deep-learning/convolutional-neural-network/fully-connected.png)

> 全连接的神经网络中，Affine 层后面跟着激活函数 ReLU 层（或者 Sigmoid 层）。
>
> 这里堆叠了 4 层 “Affine-ReLU” 组合，然后第 5 层是 Affine 层，最后由 Softmax 层输出最终结果（概率）。

那么，CNN 会是什么样的结构呢：

![基于 CNN 的网络的例子](/images/deep-learning/convolutional-neural-network/cnn.png)

> 新增了 Convolution 层和 Pooling 层（用灰色的方块表示）

CNN 中新增了 Convolution 层和 Pooling 层。CNN 的层的连接顺序是 “Convolution-ReLU-(Pooling)”（Pooling 层有时会被省略）。这可以理解为之前的 “Affine - ReLU” 连接被替换成了 “Convolution-ReLU-(Pooling)” 连接。

还需要注意的是，靠近输出的层中使用了之前的 “Affine-ReLU” 组合。此外，最后的输出层中使用了之前的 “Affine-Softmax” 组合。这些都是一般的 CNN 中比较常见的结构。

## 卷积层

### 全连接层存在的问题

::: danger 警告

该部分尚未完工!

:::

## 小结

::: danger 警告

该部分尚未完工!

:::

::: details 专有名词

- **卷积神经网络**：

- **卷积层（Convolution 层）**：

- **池化层（Pooling 层）**：

- **全链接**：神经网络中，相邻层的所有神经元之间都有连接

:::
