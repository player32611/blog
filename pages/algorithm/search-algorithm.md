# 搜索算法

::: danger 警告
该页面尚未完工!
:::

## 目录

[[toc]]

## 什么是搜索

搜索，是一种枚举，通过穷举所有的情况来找到最优解，或者统计合法解的个数。因此，搜索有时候也叫做暴搜。

搜索一般分为深度优先搜索(DFS)与宽度优先搜索(BFS)。

::: tip 回溯与剪枝

- 回溯：当在搜索的过程中，遇到走不同或者走到底的情况时，就回头。

- 剪枝：剪掉在搜索过程中，重复出现或者不是最优解的分支。

:::

## 深度优先遍历 - DFS

### 递归型枚举与回溯剪枝初识

例题：[B3622 枚举子集（递归实现指数型枚举）](https://www.luogu.com.cn/problem/B3622)

<p><font color="blue">解法：深度优先搜索</font></p>

```c++
#include<iostream>
#include<string>
using namespace std;

int n;
string path; // 记录递归过程中，每一步的决策

void dfs(int pos) {
	if (pos > n) {
		// path 就存着前 n 个人的决策
		cout << path << endl;
		return;
	}
	// 不选
	path += 'N';
	dfs(pos + 1);
	path.pop_back(); // 回溯，清空现场
	// 选
	path += 'Y';
	dfs(pos + 1);
	path.pop_back();
}

int main() {
	cin >> n;
	dfs(1);
}
```

例题：[P10448 组合型枚举](https://www.luogu.com.cn/problem/P10448)

```c++
#include<iostream>
#include<vector>
using namespace std;

int n, m;
vector<int> path;

void dfs(int begin) {
	if (path.size() == m) {
		for (auto x : path) cout << x << " ";
		cout << endl;
		return;
	}
	for (int i = begin; i <= n; i++) {
		path.push_back(i);
		dfs(i + 1);
		path.pop_back(); // 清空现场
	}
}

int main() {
	cin >> n >> m;
	dfs(1);
}
```

例题：[B3623 枚举排列（递归实现排列型枚举）](https://www.luogu.com.cn/problem/B3623)

::: danger 警告
该部分尚未完工!
:::

例题：[P1706 全排列问题](https://www.luogu.com.cn/problem/P1706)

::: danger 警告
该部分尚未完工!
:::

### DFS

例题：[P1036 [NOIP 2002 普及组] 选数](https://www.luogu.com.cn/problem/P1036)

<p><font color="blue">解法：暴力搜索</font></p>

```c++
#include<iostream>
using namespace std;

const int N = 25;

int n, k;
int a[N];
int ret;
int path; // 记录路径中所选择的数的和

bool isprime(int x) {
	if (x <= 1)return false;
	// 试除法
	for(int i=2;i<=x/i;i++){
		if (x % i == 0)return false;
	}
	return true;
}

void dfs(int pos,int begin) {
	if (pos > k) {
		if (isprime(path))ret++;
		return;
	}
	for (int i = begin; i <= n; i++) {
		path += a[i];
		dfs(pos + 1, i + 1);
		path -= a[i];
	}
}

int main() {
	cin >> n >> k;
	for (int i = 1; i <= n; i++)cin >> a[i];
	dfs(1, 1);
	cout << ret << endl;
}
```

例题：[P9241 [蓝桥杯 2023 省 B] 飞机降落](https://www.luogu.com.cn/problem/P9241)

::: danger 警告
该部分尚未完工!
:::

例题：[P1219 [USACO1.5] 八皇后 Checker Challenge](https://www.luogu.com.cn/problem/P1219)

::: danger 警告
该部分尚未完工!
:::

例题：[P1784 数独](https://www.luogu.com.cn/problem/P1784)

::: danger 警告
该部分尚未完工!
:::

### 剪枝与优化

剪枝，形象地看，就是剪掉搜索树的分支，从而减少搜索树的规模，排除掉搜索树中没有必要的分支，优化时间复杂度。

在深度优先遍历中，有几种常见的剪枝方法：

**1. 排除等效冗余**

> 如果在搜索过程中，通过某一个节点往下的若干分支中，存在最终结果等效的分支，那么就只需要搜索其中一条分支。

**2. 可行性剪枝**

> 如果在搜索过程中，发现有一条分支是无论如何都拿不到最终解，此时就可以放弃这个分支，转而搜索其它的分支。

**3. 最优性剪枝**

> 在最优化的问题中，如果在搜索过程中，发现某一个分支已经超过当前已经搜索过的最优解，那么这个分支往后的搜索，必定不会拿到最优解。此时应该停止搜索，转而搜索其它情况。

**4. 优化搜索顺序**

> 在有些搜索问题中，搜索顺序是不影响最终结果的，此时搜索顺序的不同会影响搜索树的规模。因此，应当先选择一个搜索分支规模较小的搜索顺序，快速拿到一个最优解之后，用最优性剪枝剪掉别的分支。

**5. 记忆化搜索**

> 记录每一个状态的搜索结果，当下一次搜索到这个状态时，直接找到之前记录过的搜索结果，有时也叫动态规划。

### 记忆化搜索

::: danger 警告
该部分尚未完工!
:::

## 广度优先遍历 - BFS

::: danger 警告
该部分尚未完工!
:::

## FloodFill 问题

::: danger 警告
该部分尚未完工!
:::
