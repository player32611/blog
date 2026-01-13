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

## 广度优先遍历 - BFS

::: danger 警告
该部分尚未完工!
:::

## FloodFill 问题

::: danger 警告
该部分尚未完工!
:::
