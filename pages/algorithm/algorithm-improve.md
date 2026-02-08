# 算法提高

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## 线段树

> 前置：[二叉树](data-structure.html#二叉树)、[堆](data-structure.html#堆)、[递归初阶](algorithm-basic.html#递归初阶)、[分治](algorithm-basic.html#分治)

### 引入

有如下问题：

- 有 $n(n \leq 10^5)$ 个数，$q(q \leq 10^5)$ 次操作，每次操作为询问区间 [l, r] 的和

- 有 $n(n \leq 10^5)$ 个数，$q(q \leq 10^5)$ 次操作，操作有两种：
  - 查询区间 [l, r] 的和
  - 将第 i 个数修改成 $x$

- 有 $n(n \leq 10^5)$ 个数，$q(q \leq 10^5)$ 次操作，操作有两种：
  - 查询区间 [l, r] 的和
  - 将区间 [l, r] 的数全部修改成 $x$

- 有 $n(n \leq 10^5)$ 个数，$q(q \leq 10^5)$ 次操作，每次操作为区间 [l, r] 的最大值或者最小值

这个其实是 **RMQ**（Range Minimum/Maximum Query）问题。我们使用**线段树**来解决。

线段树是一棵二叉树，常用来维护区间信息。可以在 $\log$ 级别的时间复杂度内完成：区间的单点修改，区间修改，区间查询（区间和、区间最大/最小值）等操作。

### 线段树的构建

线段树是基于分治思想的二叉树，树中的每一个结点都会维护一段区间的信息。其中叶子结点存储元素本身，非叶结点维护区间内元素的信息。

线段树类似堆的存储方式，用结构体数组来存储。

::: details 具体示例

以数组 $a=[5,1,3,0,2,2,7,4,5,8]$ 为例，如果查询的是区间和，我们会创建出来这样一棵树来维护信息：

![线段树](/images/algorithm/algorithm-improve/segment-tree.png)

:::

根据构建方式，可以得到以下性质：

- 线段树的每个结点都维护一个区间的信息；

- 线段树中的根结点维护整个区间的信息，叶子结点维护长度为 1 的区间信息；

- 可以用结构体数组来实现线段树，类似堆的存储方式，也就是二叉树的静态存储。此时父结点的编号为 $p$ 时，左孩子编号为 $p \times 2$，右孩子编号为 $p \times 2+1$；

- 若当前结点维护的区间为 [l, r]，那么左右孩子分别维护 [l, mid] 以及 [mid + 1, r] 区间的信息；

- 线段树的空间，需要开最大区间的 4 倍。

::: tip 推导过程

设数组的大小为 $n$，则：

线段树的高：$h=\log n + 2$

线段树节点的个数：$N=2^h-1=4n-1$

:::

```c++
#define lc p << 1 // 相当于 p * 2
#define rc p << 1 | 1 // 相当于 p * 2 + 1

typedef long long LL;
const int N = 1e5 + 10;

int n, m;
LL a[N];

struct node { // 线段树的结点
	LL l, r, sum;
}tr[N << 2];

// 整合左右孩子的信息
void pushup(int p) {
	tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 创建线段树
void build(int p, int l, int r) {
	tr[p] = { l,r,0 }; // 初始化
	if (l == r) { // 如果是叶子结点
		tr[p].sum = a[l]; // 更新 sum 的值
		return;
	}
	int mid = (l + r) >> 1; // 一分为二
	build(lc, l, mid); // 构建左子树
	build(rc, mid + 1, r); // 构建右子树
	pushup(p); // 左右子树构建完成之后，维护 sum 信息
}
```

时间复杂度：$O(n)$

### 区间查询

对于一个待查询的区间，用拆分 + 拼凑的思想，在线段树的结点中收集结果。具体流程：

1. 从根结点出发，向下递归；

2. 如果当前结点维护的区间信息包含在待查询的区间内，直接返回结点维护的信息；

3. 如果左区间有重叠，去左子树上找结果；

4. 如果右区间有重叠，去右子树上找结果；

```c++
// 区间查询
LL query(int p, int x, int y) {
	LL l = tr[p].l, r = tr[p].r; // 当前结点维护的信息
	if (x <= 1 && r <= y)return tr[p].sum; // 如果是查询区间的子区间，返回结果
	LL sum = 0, mid = (l + r) >> 1;
	if (x <= mid)sum += query(lc, x, y);
	if (y > mid)sum += query(rc, x, y);
	return sum;
}
```

时间复杂度：$O(\log n)$

### 单点修改

具体流程：

1. 递归找到叶子结点，并且维护修改之后的信息；

2. 然后一路向上回溯，修改所有路径上的结点信息，使得维护的信息为修改之后的信息。

::: details 具体示例

以数组 $a=[5,1,3,0,2,2,7,4,5,8]$ 为例，如果将 $x=6$ 位置上的元素增加 3，维护的信息如下：

![线段树单点修改](/images/algorithm/algorithm-improve/segment-tree-single-point-modify.png)

:::

```c++
// 单点修改
void modify(int p, int x, LL k) {
	int l = tr[p].l, r = tr[p].r;
	if (l == x && r == x) { // 找到叶子结点
		tr[p].sum += k;
		return;
	}
	int mid = (l + r) >> 1;
	if (x <= mid)modify(lc, x, k);
	else modify(rc, x, k);
	pushup(p);
}
```

[P3374 【模板】树状数组 1](https://www.luogu.com.cn/problem/P3374)

```c++
#include<iostream>
using namespace std;

#define lc p<<1
#define rc p<<1|1
typedef long long LL;

const int N = 5e5 + 10;

int n, m;
LL a[N];

struct node {
	int l, r;
	LL sum;
}tr[N << 2];

void pushup(int p) {
	tr[p].sum = tr[lc].sum + tr[rc].sum;
}

void build(int p,int l,int r) {
	tr[p] = { l,r,a[l] };
	if (l == r)return;
	int mid = (l + r) >> 1;
	build(lc, l, mid);
	build(rc, mid + 1, r);
	pushup(p);
}

void modify(int p, int x, LL k) {
	int l = tr[p].l, r = tr[p].r;
	if (x == l && r == x) {
		tr[p].sum += k;
		return;
	}
	int mid = (l + r) >> 1;
	if (x <= mid)modify(lc, x, k);
	else modify(rc, x, k);
	pushup(p);
}

LL query(int p, int x, int y) {
	int l = tr[p].l, r = tr[p].r;
	if (x <= l && r <= y)return tr[p].sum;
	int mid = (l + r) >> 1;
	LL sum = 0;
	if (x <= mid)sum += query(lc, x, y);
	if (y > mid) sum += query(rc, x, y);
	return sum;
}

int main() {
	cin >> n >> m;
	for (int i = 1; i <= n; i++)cin >> a[i];
	build(1, 1, n);
	while (m--) {
		int op, x, y;
		cin >> op >> x >> y;
		if (op == 1)modify(1, x, y);
		else cout << query(1, x, y) << endl;
	}
}
```

### 区间修改（懒标记）

::: danger 警告

该部分尚未完工!

:::

## 树状数组

::: danger 警告

该部分尚未完工!

:::

## ST 表

::: danger 警告

该部分尚未完工!

:::

## 树形 dp

::: danger 警告

该部分尚未完工!

:::

## 状压 dp

::: danger 警告

该部分尚未完工!

:::

## 数位 dp

::: danger 警告

该部分尚未完工!

:::

## 双向搜索

::: danger 警告

该部分尚未完工!

:::

## 迭代加深搜索

::: danger 警告

该部分尚未完工!

:::

## 树的重心与直径

::: danger 警告

该部分尚未完工!

:::

## 树上前缀和与差分

::: danger 警告

该部分尚未完工!

:::

## 树上倍增与 LCA

::: danger 警告

该部分尚未完工!

:::

## 差分约束

::: danger 警告

该部分尚未完工!

:::

## 同余最短路

::: danger 警告

该部分尚未完工!

:::

## 欧拉回路

::: danger 警告

该部分尚未完工!

:::

## 分层树

::: danger 警告

该部分尚未完工!

:::

## 01 分数规划

::: danger 警告

该部分尚未完工!

:::

## kmp

::: danger 警告

该部分尚未完工!

:::

## manacher

::: danger 警告

该部分尚未完工!

:::
