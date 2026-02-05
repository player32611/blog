# 算法巩固

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## 第 1 周

### Day 01

[P2669 [NOIP 2015 普及组] 金币](https://www.luogu.com.cn/problem/P2669)

<p><font color="blue">解法：模拟</font></p>

```c++
#include<iostream>
using namespace std;

int main() {
	int k;
	cin >> k;
	int ret = 0, tmp = 1, cnt = 1;
	for (int i = 1; i <= k; i++) {
		ret += tmp;
		cnt--;
		if (cnt == 0) {
			tmp++;
			cnt = tmp;
		}
	}
	cout << ret << endl;
}
```

[P1190 [NOIP 2010 普及组] 接水问题](https://www.luogu.com.cn/problem/P1190)

<p><font color="blue">解法：模拟，找出所有的水龙头中最早结束的那一个</font></p>

```c++
#include<iostream>
#include<queue>
#include<vector>
#include<functional>
#include<algorithm>
using namespace std;

int n, m;

int main() {
	cin >> n >> m;
	priority_queue<int, vector<int>, greater<int>> heap;

	for (int i = 1; i <= m; i++) {
		heap.push(0);
	}
	int ret = 0;
	for (int i = 1; i <= n; i++) {
		int x;
		cin >> x;
		int t = heap.top();
		heap.pop();
		t += x;
		heap.push(t);
		ret = max(ret, t);
	}
	cout << ret << endl;
}
```

[P1774 最接近神的人](https://www.luogu.com.cn/problem/P1774)

<p><font color="blue">解法：逆序对</font></p>

```c++
#include<iostream>
using namespace std;

typedef long long LL;
const int N = 5e5 + 10;

int n;
int a[N], tmp[N];

LL merge_sort(int l, int r) {
	if (l >= r)return 0;
	LL ret = 0;
	int mid = (l + r) / 2;
	// [l, mid] [mid + 1, r]
	ret += merge_sort(l, mid);
	ret += merge_sort(mid + 1, r);
	int cur1 = l, cur2 = mid + 1, i = l;
	while (cur1 <= mid && cur2 <= r) {
		if (a[cur1] <= a[cur2])tmp[i++] = a[cur1++];
		else {
			ret += mid - cur1 + 1;
			tmp[i++] = a[cur2++];
		}
	}
	while (cur1 <= mid)tmp[i++] = a[cur1++];
	while (cur2 <= r)tmp[i++] = a[cur2++];
	for (int j = l; j <= r; j++)a[j] = tmp[j];
	return ret;
}

int main() {
	cin >> n;
	for (int i = 1; i <= n; i++)cin >> a[i];
	cout << merge_sort(1, n) << endl;
}
```

[P1455 搭配购买](https://www.luogu.com.cn/problem/P1455)

<p><font color="blue">解法：01 背包 + bfs/dfs/并查集，将连接的物品合并为一个大物品</font></p>

```c++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

const int N = 1e4 + 10;

int n, m, w;
int c[N], d[N]; // 合并前
vector<int> edges[N];

bool st[N]; // 标记 dfs 过程中，那些点还没有标记过
int cnt;
int cc[N], dd[N]; // 合并后

int f[N];

void dfs(int a) {
	st[a] = true;
	cc[cnt] += c[a];
	dd[cnt] += d[a];
	for (auto b : edges[a]) {
		if (!st[b])dfs(b);
	}
}

int main() {
	cin >> n >> m >> w;
	for (int i = 1; i <= n; i++)cin >> c[i] >> d[i];
	for (int i = 1; i <= m; i++) {
		int a, b;
		cin >> a >> b;
		edges[a].push_back(b);
		edges[b].push_back(a);
	}
	for (int i = 1; i <= n; i++) {
		if (!st[i]) {
			cnt++;
			dfs(i);
		}
	}

	// 01 背包
	for (int i = 1; i <= cnt; i++) {
		for (int j = w; j >= cc[i]; j--)f[j] = max(f[j], f[j - cc[i]] + dd[i]);
	}
	cout << f[w] << endl;
}
```

### Day 02

[P1548 [NOIP 1997 普及组] 棋盘问题](https://www.luogu.com.cn/problem/P1548)

<p><font color="blue">解法：暴力枚举</font></p>

```c++
#include<iostream>
using namespace std;

int main() {
	int n, m;
	cin >> n >> m;
	int x = 0, y = 0;
	for (int x1 = 0; x1 <= n; x1++)
		for (int y1 = 0; y1 <= m; y1++)
			for (int x2 = x1 + 1; x2 <= n; x2++)
				for (int y2 = y1 + 1; y2 <= m; y2++) {
					int dx = x2 - x1, dy = y2 - y1;
					if (dx == dy)x++;
					else y++;
				}
	cout << x << " " << y << endl;
}
```

[P1208 [USACO1.3] 混合牛奶 Mixing Milk](https://www.luogu.com.cn/problem/P1208)

<p><font color="blue">解法：贪心策略，每次选取单价最小的</font></p>

```c++
#include<iostream>
#include<algorithm>
using namespace std;

const int N = 5010;

int n, m;

struct node {
	int p, x;
}a[N];

bool cmp(node& a, node& b) {
	return a.p < b.p;
}

int main() {
	cin >> n >> m;
	for (int i = 1; i <= m; i++)cin >> a[i].p >> a[i].x;
	sort(a + 1, a + 1 + m, cmp);
	int ret = 0, sum = 0;
	for (int i = 1; i <= m; i++) {
		int t = min(a[i].x, n - sum);
		ret += t * a[i].p;
		sum += t;
	}
	cout << ret << endl;
}
```

[P1060 [NOIP 2006 普及组] 开心的金明](https://www.luogu.com.cn/problem/P1060)

<p><font color="blue">解法：01 背包</font></p>

<p><font color="blue">状态表示：f[i][j] 表示在前 i 个物品中挑选，总价格不超过 j 的情况下，最大的价值。</font></p>

<p><font color="blue">状态转移方程：f[i][j] = max(f[i - 1][j], f[i - 1][j - v[i]] + v[i] * p[i])</font></p>

```c++
#include<iostream>
#include<algorithm>
using namespace std;

const int N = 30010, M = 30;

int n, m;
int v[M], p[M];

int f[N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++)cin >> v[i] >> p[i];
    for (int i = 1; i <= m; i++) {
        for (int j = n; j >= v[i]; j--) {
            f[j] = max(f[j], f[j - v[i]] + v[i] * p[i]);
        }
    }
    cout << f[n] << endl;
}
```

[P1083 [NOIP 2012 提高组] 借教室](https://www.luogu.com.cn/problem/P1083)

<p><font color="blue">解法：差分 + 二分答案</font></p>

```c++
#include<iostream>
using namespace std;

const int N = 1e6 + 10;

int n, m;
int r[N];
int d[N], s[N], t[N];

int f[N]; // 差分数组

// 把 [1, x] 所有的订单处理完毕之后，判断是否可行
bool check(int x) {
	// 初始差分数组
	for (int i = 1; i <= n; i++) {
		f[i] = r[i] - r[i - 1];
	}

	// 处理订单
	for (int i = 1; i <= x; i++) {
		// s[i] ~ t[i] - d[i]
		f[s[i]] -= d[i];
		f[t[i] + 1] += d[i];
	}

	for (int i = 1; i <= n; i++) {
		f[i] = f[i - 1] + f[i];
		if (f[i] < 0)return false;
	}
	return true;
}

int main() {
	cin >> n >> m;
	for (int i = 1; i <= n; i++)cin >> r[i];
	for (int i = 1; i <= m; i++)cin >> d[i] >> s[i] >> t[i];

	int l = 1, r = m;
	while (l < r) {
		int mid = (l + r) / 2;
		if (check(mid))l = mid + 1;
		else r = mid;
	}
	if (check(l))cout << 0;
	else cout << -1 << endl << l << endl;
}
```

### Day 03

[CF25B Phone numbers](https://www.luogu.com.cn/problem/CF25B)

[P2660 zzc 种田](https://www.luogu.com.cn/problem/P2660)

[P2018 消息传递](https://www.luogu.com.cn/problem/P2018)

[P6070 『MdOI R1』Decrease](https://www.luogu.com.cn/problem/P6070)

## 第 2 周

::: danger 警告

该部分尚未完工!

:::

## 第 3 周

::: danger 警告

该部分尚未完工!

:::

## 第 4 周

::: danger 警告

该部分尚未完工!

:::

## 第 5 周

::: danger 警告

该部分尚未完工!

:::

## 第 6 周

::: danger 警告

该部分尚未完工!

:::

## 第 7 周

::: danger 警告

该部分尚未完工!

:::

## 第 8 周

::: danger 警告

该部分尚未完工!

:::
