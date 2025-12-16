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

当数据的值特别大，各种类型都存不下的时候，此时就要用高精度算法来计算加减乘除：

- 先用字符串读入这个数，然后用数组逆序存储该数的每一位；

- 利用数组，模拟加减乘除运算的过程。

高精度算法本质上还是模拟算法，用代码模拟小学竖式计算加减乘除的过程。

### 高精度加法

例题：[P1601 A+B Problem（高精）](https://www.luogu.com.cn/problem/P1601)

<font color="blue">解法：模拟小学列竖式计算的过程</font>

1. 先用字符串读入，拆分每一位，逆序放在数组中

2. 利用数组，模拟小学列竖式计算加法的过程

```c++
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

const int N = 1e6 + 10;

int a[N], b[N], c[N];
int la, lb, lc;

// 高精度加法的模板 - c = a + b;
void add(int c[], int a[], int b[]) {
	for (int i = 0; i < lc; i++) {
		c[i] += a[i] + b[i]; // 对应位相加，再加上进位
		c[i + 1] = c[i] / 10; // 处理进位
		c[i] %= 10; // 处理余数
	}
	if (c[lc])lc++; // 处理边界情况（99 + 1 = 100）
}

int main() {
	string x, y;
	cin >> x >> y;
	// 1.拆分每一位，逆序放在数组中
	la = x.size();
	lb = y.size();
	lc = max(la, lb);
	for (int i = 0; i < la; i++)a[la - 1 - i] = x[i] - '0';
	for (int i = 0; i < lb; i++)b[lb - 1 - i] = y[i] - '0';
	// 2.模拟加法的过程
	add(c, a, b); // c = a + b
	// 输出结果
	for (int i = lc - 1; i >= 0; i--)cout << c[i];
}
```

### 高精度减法

例题：[P2142 高精度减法](https://www.luogu.com.cn/problem/P2142)

<font color="blue">解法：模拟小学列竖式计算的过程</font>

1. 先比较大小，然后用较大的数减去较小的数（用字符串比较之前，先比较一下长度）

2. 先用字符串读入，拆分每一位，逆序放在数组中

3. 利用数组，模拟小学列竖式计算减法过程

```c++
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

const int N = 1e6 + 10;

int a[N], b[N], c[N];
int la, lb, lc;

bool cmp(string& x, string& y) {
	// 先比较长度
	if (x.size() != y.size())return x.size() < y.size();
	// 再按照字典序的方式1比较
	return x < y;
}

void sub(int c[],int a[],int b[]) {
	for (int i = 0; i < lc; i++) {
		c[i] += a[i] - b[i]; // 对应位相减，然后处理借位
		if (c[i] < 0) {
			c[i + 1] -= 1; // 借位
			c[i] += 10;
		}
	}
	// 处理前导零
	while (lc > 1 && c[lc - 1] == 0)lc--;
}

int main() {
	string x, y;
	cin >> x >> y;
	// 1.比较大小
	if (cmp(x, y)) {
		swap(x, y);
		cout << '-';
	}
	// 2.拆分每一位，逆序放在数组中
	la = x.size();
	lb = y.size();
	lc = max(la, lb);
	for (int i = 0; i < la; i++)a[la - 1 - i] = x[i] - '0';
	for (int i = 0; i < lb; i++)b[lb - 1 - i] = y[i] - '0';
	// 3.模拟减法的过程
	sub(c, a, b); // c = a + b
	// 输出结果
	for (int i = lc - 1; i >= 0; i--)cout << c[i];
}
```

### 高精度乘法

例题：[P1303 A\*B Problem](https://www.luogu.com.cn/problem/P1303)
