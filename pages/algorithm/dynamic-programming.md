# 动态规划

::: danger 警告
该页面尚未完工!
:::

::: details 目录

[[toc]]

:::

## 入门：从记忆化搜索到动态规划

::: details 回忆：记忆化搜索

在搜索的过程中，如果搜索树中有很多重复的结点，此时可以通过一个“备忘录”，记录第一次搜索到的结果。当下一次搜索到这个结点时，直接在“备忘录”里面找结果。其中，搜索树中的一个一个结点，也称为一个一个状态。

比如经典的斐波那契数列问题：

```c++
int f[N] // 备忘录

int fib(int n) {
    // 搜索之前先往备忘录里面瞅一瞅
    if (f[n] != -1) return f[n];
    if(n == 0 || n == 1) return f[n] = n;
    // 返回之前，把结果记录在备忘录中
    f[n] = fib(n - 1) + fib(n - 2);
    return f[n];
}
```

在用记忆化搜索解决斐波那契问题时，如果关注“备忘录”的填写过程，会发现它是从左往右依次填写的。当 i 位置前面的格子填写完毕之后，就可以根据格子里面的值计算出 i 位置的值。所以，整个递归过程，我们也可以改写成循环的形式，也就是递推：

```c++
int f[N] // f[i] 表示：第 i 个斐波那契数

int fib(int n) {
    // 初始化前两个格子
    f[0] = 0;
    f[1] = 1;
    // 按照递推过程计算后面的值
    for(int i = 2;i <= n;i++) f[i] = f[i - 1]+f[i - 2];
    // 返回结果
    return f[n];
}
```

:::

动态规划（Dynamic Programming，简称 DP）是一种用于解决多阶段决策问题的算法思想。它通过将复杂问题分解为更小的子问题，并存储子问题的解（通常称为“状态”），从而避免重复计算，提高效率。因此，动态规划里，蕴含着分治与剪枝思想。

上述通过**记忆化搜索**以及**递推**解决斐波那契数列的方式，其实都是动态规划。

::: warning 注意

动态规划中的相关概念其实远远不止如此，还会有：重叠子问题、最优子结构、无后效性、有向无环图等等。

:::

在递推形式的动态规划中，常用下面的专有名词来表述：

- **状态表示**：指 `f` 数组中，每一个格子代表的含义。其中，这个数组也会称为 `dp` 数组，或者 `dp` 表。

- **状态转移方程**：指 `f` 数组中，每一个格子是如何用其余的格子推导出来的。

- **初始化**：在填表之前，根据题目中的默认条件或者问题的默认初始状态，将 `f` 数组中若干格子先填上值。

::: tip 递推形式与递归形式的对应关系

- 状态表示 <--> 递归函数的意义；

- 状态转移方程 <--> 递归函数的主函数体；

- 初始化 <--> 递归函数的递归出口。

:::

### 如何利用动态规划解决问题

**第一种方式**：记忆化搜索

- 先用递归的思想去解决问题；

- 如果有重复子问题，就改成记忆化搜索的形式。

**第二种方式**：递推形式的动态规划

1. **定义状态表示**：一般情况下根据经验 + 递归函数的意义，赋予 `dp` 数组相应的含义。

2. **推导状态转移方程**：根据状态表示以及题意，在 `dp` 表中分析，当前格子如何通过其余格子推导出来。

3. **初始化**：根据题意，先将显而易见的以及边界情况下的位置填上值。

4. **确定填表顺序**：根据状态转移方程，确定按照什么顺序来填表。

5. **确定最终结果**：根据题意，在表中找出最终结果。

例题：[P10250 [GESP样题 六级] 下楼梯](https://www.luogu.com.cn/problem/P10250)

<p><font color="blue">状态表示：f[i] 表示有 i 个台阶的时候，一共有多少种方案</font></p>

<p><font color="blue">状态转移方程：f[i] = f[i - 1] + f[i - 2] + f[i - 3]</font></p>

<p><font color="blue">初始化：保证填表是正确的、填表的时候不越界</font></p>

<p><font color="blue">填表顺序：从左到右</font></p>

<p><font color="blue">最终结果：f[n]</font></p>

```c++
#include<iostream>
using namespace std;

typedef long long LL;

const int N = 65;

int n;
LL f[N]; // f[i] 表示：有 i 个台阶的时候，一共有多少种方案

int main() {
	cin >> n;
	// 初始化
	f[0] = 1;
	f[1] = 1;
	f[2] = 2;
	for (int i = 3; i <= n; i++)f[i] = f[i - 1] + f[i - 2] + f[i - 3];
	cout << f[n] << endl;
}
```

::: tip 空间优化

由于 `f[n]` 只与 `f[i - 1]`、`f[i - 2]`、`f[i - 3]` 有关，因此可以空间优化，只存储三个变量：

```c++
#include<iostream>
using namespace std;

typedef long long LL;

const int N = 65;

int n;
LL f[N]; // f[i] 表示：有 i 个台阶的时候，一共有多少种方案

int main() {
	cin >> n;
	// 初始化
	LL a = 1, b = 1, c = 2;
	for (int i = 3; i <= n; i++) {
		LL t = a + b + c;
		a = b;
		b = c;
		c = t;
	}
	if (n == 1)cout << b << endl;
	else cout << c << endl;
}
```

但该方法很少能对时间做优化。

:::

例题：[P1216 [IOI 1994 / USACO1.5] 数字三角形 Number Triangles](https://www.luogu.com.cn/problem/P1216)

<p><font color="blue">状态表示：f[i][j] 表示从 [1, 1] 走到 [i, j] 位置时，所有方案下的最大权值</font></p>

<p><font color="blue">状态转移方程：f[i][j] = max(f[i - 1][j - 1], f[i - 1][j]) + a[i][j]</font></p>

<p><font color="blue">初始化：填表的时候不越界、保证后面的填表是正确的</font></p>

<p><font color="blue">填表顺序：从左到右，从上到下</font></p>

<p><font color="blue">最终结果：最后一行的最大值</font></p>

```c++
#include<iostream>
#include<algorithm>
using namespace std;

const int N = 1010;

int n;
int a[N][N];
int f[N][N]; // f[i][j]：从 [1, 1] 走到 [i, j] 时，所有方案下的最大权值

int main() {
	cin >> n;
	for (int i = 1; i <= n; i++)for (int j = 1; j <= i; j++)cin >> a[i][j];
	for (int i = 1; i <= n; i++)for (int j = 1; j <= i; j++)f[i][j] = max(f[i - 1][j], f[i - 1][j - 1]) + a[i][j];
	int ret = 0;
	for (int j = 1; j <= n; j++)ret = max(ret, f[n][j]);
	cout << ret << endl;
}
```

::: tip 空间优化

采用从右往左的遍历方式，可以优化空间为一维数组：

```c++
#include<iostream>
#include<algorithm>
using namespace std;

const int N = 1010;

int n;
int a[N][N];
int f[N]; // f[i][j]：从 [1, 1] 走到 [i, j] 时，所有方案下的最大权值

int main() {
	cin >> n;
	for (int i = 1; i <= n; i++)for (int j = 1; j <= i; j++)cin >> a[i][j];
	for (int i = 1; i <= n; i++)for (int j = i; j >= 1; j--)f[j] = max(f[j], f[j - 1]) + a[i][j]; //修改一下遍历顺序
	int ret = 0;
	for (int j = 1; j <= n; j++)ret = max(ret, f[j]);
	cout << ret << endl;
}
```

:::

::: details 二维数组转一维数组的注意事项

- 是否修改遍历顺序

- 删掉第一维即可

:::

## 线性 dp

线性 dp 是动态规划问题中最基础，最常见的一类问题。它的特点是**状态转移只依赖于前一个或前几个状态**，状态之间的关系是线性的，通常可以用一维或者二维数组来存储状态。

### 基础线性 dp

例题：[P1192 台阶问题](https://www.luogu.com.cn/problem/P1192)

<p><font color="blue">状态表示：f[i] 表示：到达第 i 个台阶时，一共有多少种不同的方案</font></p>

<p><font color="blue">状态转移方程：f[i] += f[i - j] (1 <= j <= k)</font></p>

<p><font color="blue">初始化：f[0] = 1</font></p>

<p><font color="blue">填表顺序：从左到右</font></p>

<p><font color="blue">最终结果：f[n]</font></p>

```c++
#include<iostream>
using namespace std;

const int N = 1e5 + 10, MOD = 1e5 + 3;

int n, k;
int f[N];

int main() {
	cin >> n >> k;
	f[0] = 1;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= k && i - j >= 0; j++) {
			f[i] = (f[i] + f[i - j]) % MOD;
		}
	}
	cout << f[n] << endl;
}
```

例题：[P1115 最大子段和](https://www.luogu.com.cn/problem/P1115)

::: danger 警告

该部分尚未完工!

:::

例题：[P5888 传球游戏](https://www.luogu.com.cn/problem/P5888)

::: danger 警告

该部分尚未完工!

:::

例题：[P1541 [NOIP 2010 提高组] 乌龟棋](https://www.luogu.com.cn/problem/P1541)

::: danger 警告

该部分尚未完工!

:::

### 路径类 dp

路径类 dp 是线性 dp 的一种，它是在一个 n × m 的矩阵中设置一个行走规则，研究从起点走到终点的方案数、最小路径和或者最大路径和等等的问题。

> 入门阶段的《数字三角形》其实就是路径类 dp。

例题：[矩阵的最小路径和](https://www.nowcoder.com/practice/38ae72379d42471db1c537914b06d48e?tpld=230&tqld=39755&ru=/exam/oj)

<p><font color="blue">状态表示：f[i][j] ：从 [1, 1] 走到 [i, j] 时，所有方案下最小路径和</font></p>

<p><font color="blue">状态转移方程：f[i][j] = min(f[i - 1][j], f[i][j - 1]) + a[i][j]</font></p>

<p><font color="blue">初始化：全部格子初始化为正无穷（0x3f3f3f3f）、f[0][1] 或 f[1][0] 初始化为0</font></p>

<p><font color="blue">填表顺序：从上往下每一行、每一行从左往右</font></p>

<p><font color="blue">最终结果：f[n][m]</font></p>

```c++
#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;

const int N = 510;

int n, m;
int f[N][N];

int main() {
	cin >> n >> m;
	// 初始化
	memset(f, 0x3f, sizeof f);
	f[0][1] = 0;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			int x;
			cin >> x;
			f[i][j] = min(f[i - 1][j], f[i][j - 1]) + x;
		}
	}
	cout << f[n][m] << endl;
}
```

例题：[P1002 [NOIP 2002 普及组] 过河卒](https://www.luogu.com.cn/problem/P1002)

::: danger 警告

该部分尚未完工!

:::

例题：[P1004 [NOIP 2000 提高组] 方格取数](https://www.luogu.com.cn/problem/P1004)

::: danger 警告

该部分尚未完工!

:::

### 经典线性 dp 问题

::: danger 警告
该部分尚未完工!
:::

## 背包问题

::: danger 警告
该部分尚未完工!
:::

### 01 背包

::: danger 警告
该部分尚未完工!
:::

### 完全背包

::: danger 警告
该部分尚未完工!
:::

### 多重背包

::: danger 警告
该部分尚未完工!
:::

### 分组背包

::: danger 警告
该部分尚未完工!
:::

### 混合背包

::: danger 警告
该部分尚未完工!
:::

### 多维费用的背包问题

::: danger 警告
该部分尚未完工!
:::

## 区间 dp

::: danger 警告
该部分尚未完工!
:::
