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
