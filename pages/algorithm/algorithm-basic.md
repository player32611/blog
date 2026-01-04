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

// 高精度加法的模板 c = a + b;
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

<font color="blue">解法：模拟列竖式计算的过程</font>

1. 先用字符串读入，拆分每一位，逆序放在数组中

2. 利用数组，模拟列竖式乘法过程（无进位相乘，然后相加，最后处理进位）

```c++
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

const int N = 1e6 + 10;

int a[N], b[N], c[N];
int la, lb, lc;

// 高精度乘法的模板 c = a * b;
void mul(int c[], int a[], int b[]) {
	for (int i = 0; i < la; i++) {
		for (int j = 0; j < lb; j++) {
			c[i + j] += a[i] * b[j];
		}
	}
	// 处理进位
	for (int i = 0; i < lc; i++) {
		c[i + 1] += c[i] / 10; // 处理进位
		c[i] %= 10; // 处理余数
	}
	// 处理前导零
	while (lc > 1 && c[lc - 1] == 0)lc--;
}

int main() {
	string x, y;
	cin >> x >> y;
	// 1.拆分每一位，逆序放在数组中
	la = x.size();
	lb = y.size();
	lc = la + lb;
	for (int i = 0; i < la; i++)a[la - 1 - i] = x[i] - '0';
	for (int i = 0; i < lb; i++)b[lb - 1 - i] = y[i] - '0';
	// 2.模拟乘法的过程
	mul(c, a, b); // c = a + b
	// 输出结果
	for (int i = lc - 1; i >= 0; i--)cout << c[i];
}
```

### 高精度除法

例题：[P1480 A/B Problem](https://www.luogu.com.cn/problem/P1480)

<font color="blue">解法：模拟列竖式计算除法过程</font>

1. 先用字符串读入，拆分每一位，逆序放在数组中

2. 利用数组，模拟列竖式除法过程

```c++
#include<iostream>
#include<string>
using namespace std;

typedef long long ll;

const int N = 1e6 + 10;

int a[N], b, c[N];
int la, lc;

// 高精度除法的模板 c = a / b （高精度 / 低精度）
void div(int c[],int a[],int b) {
	ll t = 0; // 标记每次除完之后的余数
	for (int i = la - 1; i >= 0; i--) {
		// 计算当前的被除数
		t = t * 10 + a[i];
		c[i] = t / b;
		t %= b;
	}
	// 处理前导零
	while (lc > 1 && c[lc - 1] == 0)lc--;
}

int main() {
	string x;
	cin >> x >> b;
	// 1.拆分每一位，逆序放在数组中
	la = x.size();
	lc = la;
	for (int i = 0; i < la; i++)a[la - 1 - i] = x[i] - '0';
	// 2.模拟除法的过程
	div(c, a, b); // c = a + b
	// 输出结果
	for (int i = lc - 1; i >= 0; i--)cout << c[i];
}
```

## 枚举

顾名思义，就是把所有情况全都罗列出来，然后找出符合题目要求的那一个。因此，枚举是一种纯暴力的算法。

一般情况下，枚举策略都是会超时的。此时要先根据题目的数据范围来判断暴力枚举是否可以通过。如果不行的话，就要用后面学的各种算法来进行优化（比如二分、双指针、前缀和与差分等）。

使用枚举策略是，重点思考枚举的对象（枚举什么），枚举的顺序（正序还是逆序），以及枚举的方式（普通枚举？递归枚举？二进制枚举）。

### 普通枚举

例题：[P1003 [NOIP 2011 提高组] 铺地毯](https://www.luogu.com.cn/problem/P1003)

<font color="blue">解法：枚举所有的地毯，找出最后覆盖题目中点的那个地毯即可</font>

```c++
#include<iostream>
using namespace std;

const int N = 1e4 + 10;

int n;
int a[N], b[N], g[N], k[N];
int x, y;

int find() {
	// 从后往前枚举
	for (int i = n; i >= 1; i--) {
		// 判断是否覆盖
		if (a[i] <= x && b[i] <= y && a[i] + g[i] >= x && b[i] + k[i] >= y)return i;
	}
	return  -1;
}

int main() {
	cin >> n;
	for (int i = 1; i <= n; i++)cin >> a[i] >> b[i] >> g[i] >> k[i];
	cin >> x >> y;
	cout << find() << endl;
}
```

### 二进制枚举

二进制枚举：用二进制表示中的 0/1 表示两种状态，从而达到枚举各种情况。

- 利用二进制枚举时，会用到一些位运算的知识。

- 关于用二进制中的 0/1 表示状态这种方法，会在动态规划中的状态压缩 dp 中继续使用到。

- 二进制枚举的方式也可以用递归实现。

例题：[78. 子集](https://leetcode.cn/problems/subsets/description/)

<font color="blue">解法：利用二进制枚举的方式，把所有情况都枚举出来</font>

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ret;
        int n = nums.size();
        // 枚举所有的状态
        for (int st = 0; st < (1 << n); st++) {
            // 根据 st 的状态，还原出要选的数
            vector<int> tmp; // 从当前选的子集
            for (int i = 0; i < n; i++) {
                if ((st >> i) & 1)
                    tmp.push_back(nums[i]);
            }
            ret.push_back(tmp);
        }
        return ret;
    }
};
```

例题：[P10449 费解的开关](https://www.luogu.com.cn/problem/P10449)

> 1. 每一盏灯，最多只会点一次：对于一盏灯而言，只有按或者不按两种状态
> 2. 按法的先后顺序，是不影响最终结果的：不用关心按的顺序，只用关心按了什么
> 3. 第一行的按法确定之后，后续灯的按法就跟着确定了

<font color="blue">解法：暴力枚举第一行所有的按法，并根据第一行的按法，计算出当前行以及下一行被按之后的结果，推导出下一行的按法。直到按到最后一行，然后判断所有灯是否全亮。</font>

::: tip 解法实现

1. 如何枚举出第一行所有的按法？

- 用二进制枚举所有的状态，0 ~ (1 << 5) - 1

2. 如何计算出，一共按了多少次？

- 二进制表示中，一共有多少个 1

3. 用二进制表示，来存储灯的状态

- 存的时候，把 0 -> 1，把 1 -> 0，此时，题目就从全亮变成全灭

4. 如何根据 push 这个按法，计算出当前行 a[i] 以及下一行 a[i + 1] 被按之后的状态？

- 可以用位运算的只是，快速计算出被按之后的状态

:::

```c++
#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;

const int N = 10;

int n = 5;
int a[N]; // 用二进制表示，来存储灯的状态
int t[N]; // 备份 a 数组

// 计算 x 的二进制表示中一共有多少个 1
int calc(int x) {
	int cnt = 0;
	while (x) {
		cnt++;
		x &= x - 1;
	}
	return cnt;
}

int main() {
	int T;
	cin >> T;
	while (T--) {
		// 多组测试时，一定要注意清空之前的数据
		memset(a, 0, sizeof a);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				char ch;
				cin >> ch;
				// 存成相反的
				if (ch == '0')a[i] |= 1 << j;
			}
		}
		int ret = 0x3f3f3f3f; // 统计所有合法的按法中的最小值
		// 枚举第一行所有的按法
		for (int st = 0; st < (1 << n); st++) {
			memcpy(t, a, sizeof a);
			int push = st; // 当前行的按法
			int cnt = 0; // 统计当前按法下一共按了多少次
			// 依次计算后续行的结果以及按法
			for (int i = 0; i < n; i++) {
				cnt += calc(push);
				// 修改当前行被按的结果
				t[i] = t[i] ^ push ^ (push << 1) ^ (push >> 1);
				t[i] &= (1 << n) - 1; // 清空影响
				// 修改下一行的状态
				t[i + 1] ^= push;
				// 下一行的按法
				push = t[i];
			}
			if (t[n - 1] == 0)ret = min(ret, cnt);
		}
		if (ret > 6)cout << -1 << endl;
		else cout << ret << endl;
	}
}
```

## 前缀和

前缀和与差分的核心思想是**预处理**，可以在暴力枚举的过程中，快速给出查询的结果，从而优化时间复杂度。是经典的**用空间替换时间**的做法。

例题：[【模板】前缀和](https://ac.nowcoder.com/acm/problem/226282)

<font color="blue">解法：先预处理出来一个前缀和数组 f ，f[i]表示区间[1, i]中，所有元素的和</font>

```c++
#include<iostream>
using namespace std;

typedef long long ll;

const int N = 1e5 + 10;

int n, q;
ll a[N];
ll f[N]; // 前缀和数组

int main() {
	cin >> n >> q;
	for (int i = 1; i <= n; i++)cin >> a[i];
	// 处理前缀和数组
	for (int i = 1; i <= n; i++) {
		f[i] = f[i - 1] + a[i];
	}
	// 处理 q 次询问
	while (q--) {
		int l, r;
		cin >> l >> r;
		cout << f[r] - f[l - 1] << endl;
	}
}
```

::: warning 注意

使用前缀和数组时，下标必须从 1 开始计数

:::

例题：[P1115 最大子段和](https://www.luogu.com.cn/problem/P1115)

<font color="blue">解法：利用前缀和</font>

```c++
#include<iostream>
#include<algorithm>
using namespace std;

typedef long long ll;

const int N = 2e5 + 10;

int n;
ll f[N]; // 前缀和数组

int main() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		ll x;
		cin >> x;
		f[i] = f[i - 1] + x;
	}

	ll ret = -1e20;
	ll prevmin = 0;
	for (int i = 1; i <= n; i++) {
		ret = max(ret, f[i] - prevmin);
		prevmin = min(prevmin, f[i]);
	}
	cout << ret << endl;
}
```

## 其他

### ACM 模式与核心代码模式

**ACM 模式**一般是竞赛和笔试面试常用的模式，就是只给你一个题目描述，外加输入样例和输出样例，不会给你任何的代码。此时，选手或者应聘者需要根据题目要求，自己完成如下任务：

> 1. 头文件的包含
> 2. main 函数的设计
> 3. 自己定义程序所需的变量和容器（数组、哈希表等等）
> 4. 数据的输入（根据题目叙述控制输入数据的格式）
> 5. 数据的处理（各种函数接口的设计）
> 6. 数据的输出（根据题目叙述控制输出数据的格式）

**核心代码模式**就只用实现主要功能：

> 1. 核心代码模式不需要你处理头文件，输入和输出等乱七八糟的东西，只是甩给你一个函数接口。你的任务就仅仅是完成这个函数；
> 2. 在这一个函数接口中，函数头部分会传给你需要的数据，直接使用即可；
> 3. 在你完成这个函数并且提交之后，后台会调用你所写的函数，并且根据你返回的结果测试是否正确。
