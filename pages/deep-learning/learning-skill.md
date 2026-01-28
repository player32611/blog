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

SGD是一个简单的方法，比起胡乱地搜索参数空间，也算是 “聪明” 的方法。但是，根据不同的问题，也存在比 SGD 更加聪明的方法。

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

::: danger 警告

该部分尚未完工!

:::

## 小结

::: details 专有名词

- **最优化**：神经网络的学习中寻找最优参数的过程

:::
