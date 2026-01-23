# 误差反向传播法

::: danger 警告
该页面尚未完工!
:::

::: details 目录

[[toc]]

:::

## 计算图

**计算图**将计算过程用图形表示出来。这里说的图形是数据结构图，通过多个节点和边表示（连接节点的直线称为“边”）。

计算图通过节点和箭头表示计算过程。节点用 ○ 表示，○ 中是计算的内容。将计算的中间结果写在箭头的上方，表示各个节点的计算结果从左向右传递。

::: details 问题 1

- **问题 1**： 太郎在超市买了 2 个 100 日元一个的苹果，消费税是 10%，请计算支付金额。

![基于计算图求解的问题 1 的答案](/images/deep-learning/backpropagation/question1-answer1.png)

开始时，苹果的 100 日元流到“×2”节点，变成 200 日元，然后被传递给下一个节点。接着，这个 200 日元流向“×1.1”节点，变成 220 日元。因此，从这个计算图的结果可知，答案为 220 日元。

图中把“×2”“ ×1.1”等作为一个运算整体用 ○ 括起来了，不过只用 ○ 表示乘法运算“×”也是可行的。此时，如下所示，可以将“2”和“1.1”分别作为变量“苹果的个数”和“消费税”标在 ○ 外面。

![基于计算图求解的问题 1 的答案：“苹果的个数”和“消费税”作为变量标在○外面](/images/deep-learning/backpropagation/question1-answer2.png)

:::

::: details 问题 2

- **问题 2**： 太郎在超市买了 2 个苹果、3 个橘子。其中，苹果每个 100 日元，橘子每个 150 日元。消费税是 10%，请计算支付金额。

![基于计算图求解的问题 2 的答案](/images/deep-learning/backpropagation/question2.png)

这个问题中新增了加法节点“+”，用来合计苹果和橘子的金额。构建了计算图后，从左向右进行计算。就像电路中的电流流动一样，计算结果从左向右传递。到达最右边的计算结果后，计算过程就结束了。

:::

综上，用计算图解题的情况下，需要按如下流程进行：

- 构建计算图；

- 在计算图上，从左向右进行计算。

这里的第 2 歩“从左向右进行计算”是一种正方向上的传播，简称为**正向传播**（forward propagation）。正向传播是从计算图出发点到结束点的传播。

当然也可以考虑反向（从图上看的话，就是从右向左）的传播。实际上，这种传播称为**反向传播**（backward propagation）。

### 局部计算

计算图的特征是可以通过传递“局部计算”获得最终结果。“局部”这个词的意思是“与自己相关的某个小范围”。局部计算是指，无论全局发生了什么，都能只根据与自己相关的信息输出接下来的结果。

::: details 示例

比如，在超市买了2个苹果和其他很多东西：

![买了2个苹果和其他很多东西的例子](/images/deep-learning/backpropagation/buy-many.png)

> 假设（经过复杂的计算）购买的其他很多东西总共花费 4000 日元。

这里的重点是，各个节点处的计算都是局部计算。

这意味着，例如苹果和其他很多东西的求和运算（4000 + 200 → 4200）并不关心 4000 这个数字是如何计算而来的，只要把两个数字相加就可以了。

换言之，各个节点处只需进行与自己有关的计算（在这个例子中是对输入的两个数字进行加法运算），不用考虑全局。

:::

计算图可以集中精力于局部计算。无论全局的计算有多么复杂，各个步骤所要做的就是对象节点的局部计算。虽然局部计算非常简单，但是通过传递它的计算结果，可以获得全局的复杂计算的结果。

### 计算图的优点

- 无论全局是多么复杂的计算，都可以通过局部计算使各个节点致力于简单的计算，从而简化问题；

- 利用计算图可以将中间的计算结果全部保存起来（比如，计算进行到2个苹果时的金额是 200 日元、加上消费税之前的金额 650 日元等）；

- 可以通过正向传播和反向传播高效地计算各个变量的导数值。

::: details 重审问题 1

问题1中，我们计算了购买2个苹果时加上消费税最终需要支付的金额。

这里，假设我们想知道苹果价格的上涨会在多大程度上影响最终的支付金额，即求“支付金额关于苹果的价格的导数”。设苹果的价格为 $x$，支付金额为 $L$，则相当于求 $\frac{\partial L}{\partial x}$。这个导数的值表示当苹果的价格稍微上涨时，支付金额会增加多少。

可以通过计算图的反向传播求导数：

![基于反向传播的导数的传递](/images/deep-learning/backpropagation/question1-reverse.png)

如图所示，反向传播使用与正方向相反的箭头（粗线）表示。反向传播传递“局部导数”，将导数的值写在箭头的下方。

在这个例子中，反向传播从右向左传递导数的值（1 → 1.1 → 2.2）。从这个结果中可知，“支付金额关于苹果的价格的导数”的值是 2.2。这意味着，如果苹果的价格上涨 1 日元，最终的支付金额会增加 2.2 日元（严格地讲，如果苹果的价格增加某个微小值，则最终的支付金额将增加那个微小值的 2.2 倍）

这里只求了关于苹果的价格的导数，不过“支付金额关于消费税的导数”“支付金额关于苹果的个数的导数”等也都可以用同样的方式算出来。并且，计算中途求得的导数的结果（中间传递的导数）可以被共享，从而可以高效地计算多个导数。

:::

## 链式法则

前面介绍的计算图的正向传播将计算结果正向（从左到右）传递，其计算过程是我们日常接触的计算过程，所以感觉上可能比较自然。

而反向传播将局部导数向正方向的反方向（从右到左）传递，一开始可能会让人感到困惑。传递这个局部导数的原理，是基于**链式法则**（chain rule）的。

### 计算图的反向传播

假设存在 $y = f(x)$ 的计算，这个计算的反向传播如下图所示：

![计算图的反向传播：沿着与正方向相反的方向，乘上局部导数](/images/deep-learning/backpropagation/chart-backpropagation.png)

反向传播的计算顺序是，**将信号 $E$ 乘以节点的局部导数**（$\frac{\partial y}{\partial x}$），**然后将结果传递给下一个节点**。

> 这里所说的局部导数是指正向传播中 $y=f(x)$ 的导数，也就是 y 关于 x 的导数（$\frac{\partial y}{\partial x}$）。
>
> 比如，假设 $y=f(x)=x^2$，则局部导数为 $\frac{\partial y}{\partial x}=2x$。把这个局部导数乘以上游传过来的值（本例中为 $E$），然后传递给前面的节点。

通过这样的计算，可以高效地求出导数的值，这是反向传播的要点。

### 什么是链式法则

链式法则是关于复合函数的导数的性质，定义如下：

- 如果某个函数由复合函数表示，则该复合函数的导数可以用构成复合函数的各个函数的导数的乘积表示。

::: details 什么是复合函数

复合函数是由多个函数构成的函数。比如，$z=(x+y)^2$ 是由下面两个式子构成的：

$$z=t^2$$

$$t=x+y$$

:::

### 链式法则和计算图

现在我们尝试将 $z=(x+y)^2$ 的链式法则的计算用计算图表示出来。如果用“\*\*2”节点表示平方运算的话，则计算图如下图所示：

![计算图：沿着与正方向相反的方向，乘上局部导数后传递](/images/deep-learning/backpropagation/chain-backpropagation.png)

如图所示，计算图的反向传播从右到左传播信号。反向传播的计算顺序是：先将节点的输入信号乘以节点的局部导数（偏导数），然后再传递给下一个节点。

比如，反向传播时，“\*\*2”节点的输入是 $\frac{\partial z}{\partial x}$，将其乘以局部导数 $\frac{\partial z}{\partial t}$（因为正向传播时输入是 $t$、输出是 $z$，所以这个节点的局部导数是 $\frac{\partial z}{\partial t}$），然后传递给下一个节点。

另外，上图中反向传播最开始的信号 $\frac{\partial z}{\partial z}$ 在前面的数学式中没有出现，这是因为 $\frac{\partial z}{\partial z}=1$，所以在刚才的式子中被省略了。

需要注意的是最左边的反向传播的结果。根据链式法则，$\frac{\partial z}{\partial z}\frac{\partial z}{\partial t}\frac{\partial t}{\partial x}=\frac{\partial z}{\partial t}\frac{\partial t}{\partial x}=\frac{\partial z}{\partial x}$ 成立，对应“z关于x的导数”。也就是说，反向传播是基于链式法则的。

结果如下图所示，$\frac{\partial z}{\partial x}$ 的结果为 $2(x+y)$：

![根据计算图的反向传播的结果](/images/deep-learning/backpropagation/chain-backpropagation-result.png)

## 反向传播

### 加法节点的反向传播

这里以 $z=x+y$ 为对象，观察它的反向传播。

$z=x+y$ 的导数可由下式（解析性地）计算出来：

$$ \frac{\partial z}{\partial x} = 1$$

$$ \frac{\partial z}{\partial y} = 1$$

用计算图表示的话，如下图所示（左图是正向传播，右图是反向传播）：

![加法节点的反向传播：左图是正向传播，右图是反向传播](/images/deep-learning/backpropagation/add-backpropagation.png)

反向传播将从上游传过来的导数（本例中是 $\frac{\partial L}{\partial z}$）乘以 1，然后传向下游。也就是说，因为加法节点的反向传播只乘以1，所以输入的值会原封不动地流向下一个节点。

::: tip 为什么将上游传过来的导数的值设为 $\frac{\partial L}{\partial z}$

这是因为，如下图所示，我们假定了一个最终输出值为 $L$ 的大型计算图。$z=x+y$ 的计算位于这个大型计算图的某个地方，从上游会传来 $\frac{\partial L}{\partial z}$ 的值，并向下游传递 $\frac{\partial L}{\partial x}$ 和 $\frac{\partial L}{\partial y}$。

![反向传播的计算顺序](/images/deep-learning/backpropagation/large-computational-graph.png)

:::

::: details 具体示例

假设有 `10 + 5 = 15` 这一计算，反向传播时，从上游会传来值 1.3。用计算图表示的话如下：

![加法节点的反向传播](/images/deep-learning/backpropagation/add-backpropagation-example.png)

因为加法节点的反向传播只是将输入信号输出到下一个节点，所以反向传播将 1.3 向下一个节点传递。

:::

### 乘法节点的反向传播

这里我们考虑 $z=xy$。这个式子的导数用下式表示：

$$ \frac{\partial z}{\partial x} = y$$

$$ \frac{\partial z}{\partial y} = x$$

乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”后传递给下游。翻转值表示一种翻转关系，正向传播时信号是 x 的话，反向传播时则是 y；正向传播时信号是 y 的话，反向传播时则是 x。

![乘法节点的反向传播](/images/deep-learning/backpropagation/multiply-backpropagation.png)

::: details 具体示例

假设有 `10 × 5 = 50` 这一计算，反向传播时，从上游会传来值 1.3。用计算图表示的话，如下图所示：

![乘法节点的反向传播](/images/deep-learning/backpropagation/multiply-backpropagation-example.png)

因为乘法的反向传播会乘以输入信号的翻转值，所以各自可按 `1.3 × 5 = 6.5`、`1.3 × 10 = 13` 计算。

:::

::: warning 注意

加法的反向传播只是将上游的值传给下游，并不需要正向传播的输入信号。

但是，乘法的反向传播需要正向传播时的输入信号值。因此，实现乘法节点的反向传播时，要保存正向传播的输入信号。

:::

### 苹果的例子

再来思考一下最开始举的购买苹果的例子（2个苹果和消费税）。

这里要解的问题是苹果的价格、苹果的个数、消费税这 3 个变量各自如何影响最终支付的金额。这个问题相当于求“支付金额关于苹果的价格的导数”“支付金额关于苹果的个数的导数”“支付金额关于消费税的导数”。

用计算图的反向传播来解的话，求解过程如下图所示：

![求解苹果的导数](/images/deep-learning/backpropagation/apple-backpropagation.png)

从上图的结果可知，苹果的价格的导数是 2.2，苹果的个数的导数是 110，消费税的导数是 200。

这可以解释为，如果消费税和苹果的价格增加相同的值，则消费税将对最终价格产生 200 倍大小的影响，苹果的价格将产生 2.2 倍大小的影响。

> 不过，因为这个例子中消费税和苹果的价格的量纲不同，所以才形成了这样的结果（消费税的 1 是 100%，苹果的价格的 1 是 1 日元）。

## 简单层的实现

这里，我们把要实现的计算图的乘法节点称为“乘法层”（MulLayer），加法节点称为“加法层”（AddLayer）。

### 乘法层的实现

层的实现中有两个共通的方法（接口）`forward()` 和 `backward()`。`forward()` 对应正向传播，`backward()` 对应反向传播。

```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y # 翻转 x 和 y
        dy = dout * self.x

        return dx, dy
```

::: details 代码解释

`__init__()` 中会初始化实例变量 x 和 y，它们用于保存正向传播时的输入值。

`forward()` 接收 x 和 y 两个参数，将它们相乘后输出。

`backward()` 将从上游传来的导数（dout）乘以正向传播的翻转值，然后传给下游。

:::

![购买2个苹果](/images/deep-learning/backpropagation/apple-backpropagation.png)

使用这个乘法层的话，苹果的例子的正向传播可以像下面这样实现：

```python
apple = 100
apple_num = 2
tax = 1.1

# layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price)) # 220
print("dApple:", dapple) # 2.2
print("dApple_num:", int(dapple_num)) # 110
print("dTax:", dtax) # 200
```

::: warning 注意

这里，调用 `backward()` 的顺序与调用 `forward()` 的顺序相反。

此外，要注意 `backward()` 的参数中需要输入“关于正向传播时的输出变量的导数”。比如，`mul_apple_layer` 乘法层在正向传播时会输出 `apple_price`，在反向传播时，则会将 `apple_price` 的导数 `dapple_price` 设为参数。

:::

### 加法层的实现

接下来，我们实现加法节点的加法层:

```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```

::: details 代码解释

加法层不需要特意进行初始化，所以 `__init__()` 中什么也不运行（`pass` 语句表示“什么也不运行”）。

`forward()` 接收 `x` 和 `y` 两个参数，将它们相加后输出。

`backward()` 将上游传来的导数（`dout`）原封不动地传递给下游。

:::

### 例子

现在，我们使用加法层和乘法层，实现购买2个苹果和3个橘子的例子：

![购买2个苹果和3个橘子](/images/deep-learning/backpropagation/buy-apple-orange.png)

```python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price)) # 715
print("dApple:", dapple) # 2.2
print("dApple_num:", int(dapple_num)) # 110
print("dOrange:", dorange) # 3.3
print("dOrange_num:", int(dorange_num)) # 165
print("dTax:", dtax) # 650
```

::: details 代码解释

首先，生成必要的层，以合适的顺序调用正向传播的 `forward()` 方法。

然后，用与正向传播相反的顺序调用反向传播的 `backward()` 方法，就可以求出想要的导数。

:::

## 激活函数层的实现

现在，我们将计算图的思路应用到神经网络中。这里，我们把构成神经网络的层实现为一个类。先来实现激活函数的 ReLU 层和 Sigmoid 层。

### ReLU 层

::: details 回忆：激活函数 ReLU

ReLU 函数在输入大于 0 时，直接输出该值；在输入小于等于 0 时，输出 0。

$$
y =\begin{cases}
x & (x > 0) \\
0 & (x \leq 0)
\end{cases}
$$

:::

可以求出y关于x的导数：

$$
\frac{\partial y}{\partial x} =\begin{cases}
1 & (x > 0) \\
0 & (x \leq 0)
\end{cases}
$$

在该式中，如果正向传播时的输入 `x` 大于 0，则反向传播会将上游的值原封不动地传给下游。反过来，如果正向传播时的 `x` 小于等于 0，则反向传播中传给下游的信号将停在此处。用计算图表示的话，如下图所示：

![ReLU](/images/deep-learning/backpropagation/relu-multiply.png)

现在我们来实现 ReLU 层（在神经网络的层的实现中，一般假定 `forward()` 和 `backward()` 的参数是 NumPy 数组。）：

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # 将负数或零的位置设为 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0 # 将之前输入为负或零的位置的梯度设为 0
        dx = dout
        return dx
```

::: details 代码解释

Relu 类有实例变量 `mask`。这个变量 `mask` 是由 `True/False` 构成的 NumPy 数组，它会把正向传播时的输入 `x` 的元素中小于等于 0 的地方保存为 `True`，其他地方（大于 0 的元素）保存为 `False`。

如下例所示，`mask` 变量保存了由 `True/False` 构成的 NumPy 数组：

```python
x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
print(x)
# [[ 1.  -0.5]
# [-2.   3. ]]
mask = (x <= 0)
print(mask)
# [[False  True]
# [ True False]]
```

如果正向传播时的输入值小于等于 0，则反向传播的值为 0。因此，反向传播中会使用正向传播时保存的 `mask`，将从上游传来的 `dout` 的 `mask` 中的元素为 `True` 的地方设为 0。

:::

### Sigmoid 层

::: details 回忆：激活函数 Sigmoid

Sigmoid 函数将输入映射到 0 到 1 之间：

$$y = \frac{1}{1 + e^{-x}}$$

:::

用计算图表示的话，如下图所示：

![sigmoid层的计算图（仅正向传播）](/images/deep-learning/backpropagation/sigmoid-multiply-forward.png)

这里除了 **×** 和 **+** 节点外，还出现了新的 **exp** 和 **/** 节点。**exp** 节点会进行 $y=e^x$ 的计算，**/** 节点会进行 $y=\frac{1}{x}$ 的计算。

下面我们就来进行该计算图的反向传播：

- **步骤一**：**/** 结点表示 $\frac{1}{x}$，它的导数可以解析性地表示为 $\frac{\partial y}{\partial x} = -\frac{1}{x^2} = -y^2$。反向传播时，会将上游的值乘以 $−y^2$（正向传播的输出的平方乘以 −1 后的值）后，再传给下游。

![sigmoid层的计算图（反向传播）-步骤一](/images/deep-learning/backpropagation/sigmoid-multiply-backward-step1.png)

- **步骤二**：**+** 节点将上游的值原封不动地传给下游。

![sigmoid层的计算图（反向传播）-步骤二](/images/deep-learning/backpropagation/sigmoid-multiply-backward-step2.png)

- **步骤三**：**exp** 节点表示 $y=e^x$，它的导数表示为 $\frac{\partial y}{\partial x} = e^x$。计算图中，上游的值乘以正向传播时的输出（这个例子中是 $e^x$）后，再传给下游。

![sigmoid层的计算图（反向传播）-步骤三](/images/deep-learning/backpropagation/sigmoid-multiply-backward-step3.png)

- **步骤四**：**×** 节点将正向传播时的值翻转后做乘法运算。因此，这里要乘以 −1。

![sigmoid层的计算图](/images/deep-learning/backpropagation/sigmoid-multiply-backward.png)

由结果可知：反向传播的输出为 $\frac{\partial L}{\partial y}y^2 e^{-x}$，这个值会传播给下游的节点。这里要注意，$\frac{\partial L}{\partial y}y^2 e^{-x}$ 这个值只根据正向传播时的输入 $x$ 和输出 $y$ 就可以算出来。因此，可以画成集约化的 **sigmoid** 节点：

![Sigmoid层的计算图（简洁版）](/images/deep-learning/backpropagation/sigmoid-multiply-backward-compact.png)

简洁版的计算图可以省略反向传播中的计算过程，因此计算效率更高。此外，通过对节点进行集约化，可以不用在意Sigmoid层中琐碎的细节，而只需要专注它的输入和输出，这一点也很重要。

::: tip 进一步整理

$\frac{\partial L}{\partial y}y^2 e^{-x}$ 可以进一步整理如下：

$$
\frac{\partial L}{\partial y}y^2 e^{-x}=\frac{\partial L}{\partial y}\frac{1}{(1+e^{-x})^2}e^{-x}
=\frac{\partial L}{\partial y}\frac{1}{1+e^{-x}}\frac{e^{-x}}{1+e^{-x}}
=\frac{\partial L}{\partial y}y(1-y)
$$

因此，Sigmoid 层的反向传播，只根据正向传播的输出就能计算出来：

![Sigmoid层的计算图：可以根据正向传播的输出y计算反向传播](/images/deep-learning/backpropagation/sigmoid-multiply-backward-optimization.png)
:::

现在，我们用 Python 实现 Sigmoid 层：

```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```

::: details 代码解释

这个实现中，正向传播时将输出保存在了实例变量 `out` 中。然后，反向传播时，使用该变量 `out` 进行计算。

:::

## Affine/Softmax 层的实现

### Affine 层

神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”。因此，这里将进行仿射变换的处理实现为 **Affine 层**。

神经网络的正向传播中，为了计算加权信号的总和，使用了矩阵的乘积运算（NumPy 中是 `np.dot()`）。

```python
X = np.random.rand(2) # 输入
W = np.random.rand(2,3) # 权重
B = np.random.rand(3) # 偏置

X.shape # (2,)
W.shape # (2, 3)
B.shape # (3,)

Y = np.dot(X, W) + B
```

这里，`X`、`W`、`B` 分别是形状为(2,)、(2,3)、(3,)的多维数组。这样一来，神经元的加权和可以用 `Y = np.dot(X, W) + B` 计算出来。然后，`Y` 经过激活函数转换后，传递给下一层。这就是神经网络正向传播的流程。

现在将这里进行的求矩阵的乘积与偏置的和的运算用计算图表示出来（将乘积运算用 **dot** 节点表示）：

![Affine层的计算图（仅正向传播）](/images/deep-learning/backpropagation/affine-multiply.png)

::: warning 注意

之前我们见到的计算图中各个节点间流动的是标量，而这个例子中各个节点间传播的是矩阵。

:::

现在我们来考虑该计算图的反向传播。以矩阵为对象的反向传播，按矩阵的各个元素进行计算时，步骤和以标量为对象的计算图相同。实际写一下的话，可以得到下式：

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y} · W^T$$

$$\frac{\partial L}{\partial W}=X^T · \frac{\partial L}{\partial Y}$$

现在，我们根据上式，尝试写出计算图的反向传播：

![Affine层的反向传播：注意变量是多维数组。反向传播时各个变量的下方标记了该变量的形状](/images/deep-learning/backpropagation/affine-multiply-backward.png)

我们看一下计算图中各个变量的形状。尤其要注意，$X$ 和 $\frac{\partial L}{\partial X}$ 形状相同，$W$ 和 $\frac{\partial L}{\partial W}$ 形状相同。从下面的数学式可以很明确地看出 $X$ 和 $\frac{\partial L}{\partial X}$ 形状相同：

$$X=(x_0,x_1,···,x_n)$$

$$X^T=(\frac{\partial L}{\partial x_0},\frac{\partial L}{\partial x_1},···,\frac{\partial L}{\partial x_n})$$

::: details 为什么要注意矩阵的形状

因为矩阵的乘积运算要求对应维度的元素个数保持一致，通过确认一致性。比如，$\frac{\partial L}{\partial Y}$ 的形状是 (3,)，$W$ 的形状是 (2,3) 时，思考 $\frac{\partial L}{\partial Y}$ 和 $W$ 的乘积，使得 $\frac{\partial L}{\partial X}$ 的形状为 (2,)。

![矩阵的乘积（dot 节点）的反向传播可以通过组建使矩阵对应维度的元素个数一致的乘积运算而推导出来](/images/deep-learning/backpropagation/matrix-multiply-backward.png)

:::

### 批版本的 Affine 层

前面介绍的 Affine 层的输入 X 是以单个数据为对象的。现在我们考虑 N 个数据一起进行正向传播的情况，也就是批版本的 Affine 层。

先给出批版本的 Affine 层的计算图：

![批版本的Affine层的计算图](/images/deep-learning/backpropagation/affine-multiply-batch.png)

与刚刚不同的是，现在输入 X 的形状是(N,2)。之后就和前面一样，在计算图上进行单纯的矩阵计算。反向传播时，如果注意矩阵的形状，就可以和前面一样推导出 $\frac{\partial L}{\partial X}$ 和 $\frac{\partial L}{\partial W}$。

加上偏置时，需要特别注意。正向传播时，偏置被加到 $X·W$ 的各个数据上。

::: details 示例：加上偏置

N=2（数据为2个）时，偏置会被分别加到这 2 个数据（各自的计算结果）上：

```python
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print(X_dot_W)
# [[ 0  0  0]
#  [10 10 10]]
print(X_dot_W+B)
# [[ 1  2  3]
#  [11 12 13]]
```

正向传播时，偏置会被加到每一个数据（第1个、第2个······）上。因此，反向传播时，各个数据的反向传播的值需要汇总为偏置的元素：

```python
dY = np.array([[1, 2, 3,], [4, 5, 6]])
print(dY)
# [[1 2 3]
#  [4 5 6]]
dB = np.sum(dY, axis=0)
print(dB)
# [5 7 9]
```

这个例子中，假定数据有 2 个（N=2）。偏置的反向传播会对这 2 个数据的导数按元素进行求和。因此，这里使用了 `np.sum()` 对第 0 轴（以数据为单位的轴，axis=0）方向上的元素进行求和。

:::

综上所述，Affine 的实现如下所示：

```python
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```

### Softmax-with-Loss 层
