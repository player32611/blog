# 与学习相关的技巧

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## 参数的更新

神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为**最优化**（optimization）。但遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。

之前我们为了找到最优参数，我们将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为**随机梯度下降法**（stochastic gradient descent），简称 **SGD**。

SGD 是一个简单的方法，比起胡乱地搜索参数空间，也算是 “聪明” 的方法。但是，根据不同的问题，也存在比 SGD 更加聪明的方法。

### SGD

用数学式可以将 SGD 写成如下的式：

![SGD数学式](/images/deep-learning/learning-skill/SGD.png)

> $W$：需要更新的权重参数
>
> $\frac{\partial L}{\partial W}$：损失函数关于 $W$ 的梯度
>
> $\eta$：学习率，实际上会取 0.01 或 0.001 这些事先决定好的值
>
> ←：表示用右边的值更新左边的值

可知，SGD 是朝着梯度方向只前进一定距离的简单方法。现在，我们将 SGD 实现为一个 Python 类（为方便后面使用，我们将其实现为一个名为 SGD 的类）：

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

::: details 代码解释

这里，进行初始化时的参数 `lr` 表示 learning rate（学习率）。这个学习率会保存为实例变量。

代码段中还定义了 `update(params, grads)` 方法，这个方法在 SGD 中会被反复调用。

参数 `params` 和 `grads`（与之前的神经网络的实现一样）是字典型变量，按 `params['W1']`、`grads['W1']` 的形式，分别保存了权重参数和它们的梯度。

:::

使用这个 SGD 类，可以按如下方式进行神经网络的参数的更新（下面的代码是不能实际运行的伪代码）。

```
network = TwoLayerNet(...)
optimizer = SGD()
for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...) # mini-batch
    grads = network.gradient(x_batch, t_batch)
    params = network.params
optimizer.update(params, grads)
```

::: details 伪代码解释

这里首次出现的变量名 `optimizer` 表示 “进行最优化的人” 的意思，这里由 SGD 承担这个角色。参数的更新由 optimizer 负责完成。我们在这里需要做的只是将参数和梯度的信息传给 `optimizer`。

:::

像这样，通过单独实现进行最优化的类，功能的模块化变得更简单。这样一来，只需要将 `SGD()` 换成其它可行的优化类，就可以实现快速切换。

### SGD 的缺点

虽然SGD简单，并且容易实现，但是在解决某些问题时可能没有效率。

::: details 具体示例

我们来思考一下求下面这个函数的最小值的问题：

$$ f(x, y) = \frac{1}{20}x^2 + y^2 $$

![图形（左图）和等高线（右图）](/images/deep-learning/learning-skill/SGD-example.png)

> $f(x, y) = \frac{1}{20}x^2 + y^2$ 的图形（左图）和它的等高线（右图）

该函数数是向 x 轴方向延伸的 “碗” 状函数，且等高线呈向 x 轴方向延伸的椭圆状。

如果用图表示梯度的话，则如下所示：

![梯度图](/images/deep-learning/learning-skill/SGD-gradient.png)

这个梯度的特征是，y 轴方向上大，x 轴方向上小。换句话说，就是 y 轴方向的坡度大，而 x 轴方向的坡度小。这里需要注意的是，虽然这个函数的最小值在 (x,y)=(0,0) 处，但是图中的梯度在很多地方并没有指向(0,0)。

我们来尝试对这种形状的函数应用 SGD。从 (x,y)=(−7.0,2.0) 处（初始值）开始搜索：

![基于SGD的最优化的更新路径：呈 “之” 字形朝最小值(0,0)移动，效率低](/images/deep-learning/learning-skill/SGD-search.png)

结果显示，SGD 呈 “之” 字形移动。这是一个相当低效的路径。也就是说，SGD 的缺点是，如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。

:::

因此，我们需要比单纯朝梯度方向前进的 SGD 更聪明的方法。SGD 低效的根本原因是，梯度的方向并没有指向最小值的方向。

为了改正 SGD 的缺点，下面我们将介绍 **Momentum**、**AdaGrad**、**Adam** 这 3 种方法来取代 SGD。

### Momentum

**Momentum**是 “动量” 的意思，和物理有关。用数学式表示 Momentum 方法，如下所示：

![Momentum 数学式](/images/deep-learning/learning-skill/Momentum.png)

> $W$：需要更新的权重参数
>
> $\frac{\partial L}{\partial W}$：损失函数关于 $W$ 的梯度
>
> $\eta$：学习率
>
> $v$：对应物理上的速度
>
> ←：表示用右边的值更新左边的值
>
> $a$：承担逐渐减速的任务

该式表示了物体在梯度方向上受力，在这个力的作用下，物体的速度增加这一物理法则。

::: details 具体示例

Momentum 方法给人的感觉就像是小球在地面上滚动：

![小球滚动](/images/deep-learning/learning-skill/Momentum-ball.png)

:::

同时式中有 $αv$ 这一项。在物体不受任何力时，该项承担使物体逐渐减速的任务（$α$ 设定为 0.9 之类的值），对应物理上的地面摩擦或空气阻力。

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
```

::: details 代码解释

实例变量 `v` 会保存物体的速度。初始化时，`v` 中什么都不保存，但当第一次调用 `update()` 时，`v` 会以字典型变量的形式保存与参数结构相同的数据。

:::

现在尝试使用 Momentum 解决函数 $f(x, y) = \frac{1}{20}x^2 + y^2$ 的最优化问题：

![基于 Momentum 的最优化的更新路径](/images/deep-learning/learning-skill/Momentum-search.png)

可以看到，更新路径就像小球在碗中滚动一样。

和 SGD 相比，我们发现 “之” 字形的 “程度” 减轻了。这是因为虽然 x 轴方向上受到的力非常小，但是一直在同一方向上受力，所以朝同一个方向会有一定的加速。反过来，虽然 y 轴方向上受到的力很大，但是因为交互地受到正方向和反方向的力，它们会互相抵消，所以 y 轴方向上的速度不稳定。

因此，和 SGD 时的情形相比，可以更快地朝 x 轴方向靠近，减弱 “之” 字形的变动程度。

### AdaGrad

在神经网络的学习中，学习率（数学式中记为 $η$）的值很重要。学习率过小，会导致学习花费过多时间；反过来，学习率过大，则会导致学习发散而不能正确进行。

在关于学习率的有效技巧中，有一种被称为**学习率衰减**（learning ratedecay）的方法，即随着学习的进行，使学习率逐渐减小。实际上，一开始 “多” 学，然后逐渐 “少” 学的方法，在神经网络的学习中经常被使用。

逐渐减小学习率的想法，相当于将 “全体” 参数的学习率值一起降低。而 **AdaGrad** 进一步发展了这个想法，针对 “一个一个” 的参数，赋予其 “定制” 的值。

AdaGrad 会为参数的每个元素适当地调整学习率，与此同时进行学习（AdaGrad 的 Ada 来自英文单词 Adaptive，即 “适当的” 的意思）。

![AdaGrad 数学式](/images/deep-learning/learning-skill/AdaGrad.png)

> $W$：需要更新的权重参数
>
> $\frac{\partial L}{\partial W}$：损失函数关于 $W$ 的梯度
>
> $\eta$：学习率
>
> $h$：保存了以前的所有梯度值的平方和
>
> ←：表示用右边的值更新左边的值
>
> ʘ：对应矩阵元素的乘法

在更新参数时，通过乘以 $\frac{1}{\sqrt{h}}$，就可以调整学习的尺度。这意味着，参数的元素中变动较大（被大幅更新）的元素的学习率将变小。也就是说，可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。

::: warning 注意

AdaGrad 会记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小。实际上，如果无止境地学习，更新量就会变为 0，完全不再更新。为了改善这个问题，可以使用 RMSProp 方法。RMSProp 方法并不是将过去所有的梯度一视同仁地相加，而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反映出来。这种操作从专业上讲，称为 “指数移动平均”，呈指数函数式地减小过去的梯度的尺度。

:::

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

::: details 代码解释

这里需要注意的是，最后一行加上了微小值 `1e-7`。这是为了防止当 `self.h[key]` 中有 0 时，将 0 用作除数的情况。在很多深度学习的框架中，这个微小值也可以设定为参数，但这里我们用的是 `1e-7` 这个固定值。

:::

现在，让我们试着使用 AdaGrad 解决函数 $f(x, y) = \frac{1}{20}x^2 + y^2$ 的最优化问题：

![基于 AdaGrad 的最优化的更新路径](/images/deep-learning/learning-skill/AdaGrad-search.png)

可知，函数的取值高效地向着最小值移动。由于 y 轴方向上的梯度较大，因此刚开始变动较大，但是后面会根据这个较大的变动按比例进行调整，减小更新的步伐。因此，y 轴方向上的更新程度被减弱，“之” 字形的变动程度有所衰减。

### Adam

Momentum 参照小球在碗中滚动的物理规则进行移动，AdaGrad 为参数的每个元素适当地调整更新步伐。如果将这两个方法融合在一起会怎么样呢？这就是 **Adam** 方法的基本思路。

Adam 是 2015 年提出的方法。它的理论有些复杂，直观地讲，就是融合了 Momentum 和 AdaGrad 的方法。通过组合前面两个方法的优点，有望实现参数空间的高效搜索。此外，进行超参数的 “偏置校正” 也是 Adam 的特征。

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

现在尝试使用 Adam 解决函数 $f(x, y) = \frac{1}{20}x^2 + y^2$ 的最优化问题：

![基于 Adam 的最优化的更新路径](/images/deep-learning/learning-skill/Adam-search.png)

基于 Adam 的更新过程就像小球在碗中滚动一样。虽然 Momentun 也有类似的移动，但是相比之下，Adam 的小球左右摇晃的程度有所减轻。这得益于学习的更新程度被适当地调整了。

::: tip 提示

Adam 会设置 3 个超参数。一个是学习率（论文中以 $α$ 出现），另外两个是一次 momentum 系数 $β_1$ 和二次 momentum 系数 $β_2$。根据论文，标准的设定值是 $β_1$ 为 0.9，$β_2$ 为 0.999。设置了这些值后，大多数情况下都能顺利运行。

:::

### 使用哪种更新方法呢

这里我们来比较一下 SGD、Momentum、AdaGrad、Adam 这 4 种方法：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # for simple contour line
    mask = Z > 7
    Z[mask] = 0

    # plot
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.show()
```

![最优化方法的比较：SGD、Momentum、AdaGrad、Adam](/images/deep-learning/learning-skill/optimizer-compare.png)

根据使用的方法不同，参数更新的路径也不同。只看这个图的话，AdaGrad 似乎是最好的，不过也要注意，结果会根据要解决的问题而变。并且，很显然，超参数（学习率等）的设定值不同，结果也会发生变化。

这 4 种方法各有各的特点，都有各自擅长解决的问题和不擅长解决的问题。很多研究中至今仍在使用 SGD。Momentum 和 AdaGrad 也是值得一试的方法。最近，很多研究人员和技术人员都喜欢用 Adam。

::: details 基于MNIST数据集的更新方法的比较

我们以手写数字识别为例，比较前面介绍的 SGD、Momentum、AdaGrad、Adam 这 4 种方法，并确认不同的方法在学习进展上有多大程度的差异。

```python
# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0:读入MNIST数据==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1:进行实验的设置==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []


# 2:开始训练==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.绘制图形==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
```

> 这个实验以一个 5 层神经网络为对象，其中每层有 100 个神经元。激活函数使用的是 ReLU。

![基于MNIST数据集的更新方法的比较](/images/deep-learning/learning-skill/optimizer-compare-mnist.png)

从图中结果可知，与 SGD 相比，其他 3 种方法学习得更快，而且速度基本相同，仔细看的话，AdaGrad 的学习进行得稍微快一点。这个实验需要注意的地方是，实验结果会随学习率等超参数、神经网络的结构（几层深等）的不同而发生变化。不过，一般而言，与 SGD 相比，其他 3 种方法可以学习得更快，有时最终的识别精度也更高。

:::

## 小结

::: details 专有名词

- **最优化**：神经网络的学习中寻找最优参数的过程

- **SGD（随机梯度下降法）**：一种最优化方法，使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数

- **Momentum**：一种改进随机梯度下降法的方法，在梯度方向上受力，使参数的更新更平滑

- **AdaGrad**：一种改进随机梯度下降法，会为参数的每个元素适当地调整学习率，与此同时进行学习

- **Adam**：一种改进随机梯度下降法，融合了 Momentum 和 AdaGrad 的优点，并添加了超参数的 “偏置校正”

:::
