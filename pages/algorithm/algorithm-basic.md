# 算法基础

::: danger 警告
该页面尚未完工!
:::

## 目录

[[toc]]

## 模拟

模拟，顾名思义，就是题目让你做什么你就做什么，考察的是将思路转化成代码的代码能力。

这类题一般较为简单，属于竞赛里面的签到题。

### 多项式输出

例题：[P1067 [NOIP 2009 普及组] 多项式输出](https://www.luogu.com.cn/problem/P1067)

### 蛇形方阵

例题：[P5731 【深基 5.习 6】蛇形方阵](https://www.luogu.com.cn/problem/P5731)

**通用的解法-解决矩阵中填数的题目**

1. 定义方向向量

2. 根据规则结合方向向量填数（朝一个方向走，一边走一边填数，直到越界；越界之后，结合方向向量，重新计算出新的坐标以及方向）

```c++
#include<cstdio>

const int N = 15;

// 定义 右、下、左、上 四个方向
int dx[] = { 0,1,0,-1 };
int dy[] = { 1,0,-1,0 };

int arr[N][N];

int main() {
	int n;
	scanf_s("%d", &n);
	// 模拟填数过程
	int x = 1, y = 1; // 初始位置
	int cnt = 1; // 当前位置要填的数
	int pos = 0; // 当前的方向
	while (cnt <= n * n) {
		arr[x][y] = cnt;
		// 计算下一个位置
		int a = x + dx[pos], b = y + dy[pos];
		// 判断是否越界
		if (a<1 || a>n || b<1 || b>n || arr[a][b]) {
			// 更新出正确的该走的位置
			pos = (pos + 1) % 4;
			a = x + dx[pos], b = y + dy[pos];
		}
		x = a, y = b;
		cnt++;
	}
	// 输出
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++)printf("%3d", arr[i][j]);
		puts("");
	}
}
```

## 高精度
