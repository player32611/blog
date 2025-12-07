# STL

::: danger 警告
该页面尚未完工!
:::

## 目录

[[toc]]

## 什么是 STL

STL 即标准模板库(Standard Template Library)，是 C++标准库的一部分，里面包含了一些模板化的通用的数据结构和算法。由于其模板化的特点，它能够兼容自定义的数据类型，避免大量的造轮子工作。

NOI 和 ICPC 赛事都支持 STL 库的使用，蓝桥杯也是支持的。因此，一定要学习 STL 的使用，能够极大的提高编写代码的效率。

STL 的实现涉及比较高深的 C++ 知识，比如类、模板、容器适配等。如果想搞清楚背后的原理，就必须要学会这些知识。仅需做到：对于 STL 中一些的数据结构和算法，知道有这个东西，并且知道如何使用，以及会有什么效果。

## 动态顺序表-vector

如果需要使用动态顺序表，C++ 的 STL 提供了一个已经封装好的容器-`vector`。有的地方也叫做可变长的数组。`vector` 的底层就是一个会自动扩容的顺序表，其中创建以及增删改查等等的逻辑已经实现好了，并且也完成了封装。

### 创建 vector

```c++
#include<vector>

int main() {
	std::vector<int> a1;
    std::vector<int> a2(10);
    std::vector<int> a3(10, 2);
    std::vector<int> a4 = { 1,2,3,4,5 };
}
```

以上代码创建了:

- 一个名字为 a1 的可变长数组，里面都是 int 类型的数据；

- 一个名字为 a2 的可变长数组，大小为 10；

- 一个名字为 a3 的可变长数组，大小为 10，里面的值都初始化为 2。

- 一个名字为 a4 的可变长数组，里面有 5 个数据，数据初始化为 1,2,3,4,5。

::: tip 提示

`<>`里面可以存放任意的数据类型，包括结构体 struct、字符串 string、顺序表 vector 等等。

:::

::: warning 注意

`vector<int> a[N]`创建了一个大小为 N 的 vector 数组。

:::

### size()/empty()

- `size()`: 返回实际元素的个数；

- `empty()`: 返回顺序表是否为空。如果为空：返回 true，否则返回 false。

```c++
#include<iostream>
#include<vector>
using namespace std;

void print(vector<int>& a) {
	for (int i = 0; i < a.size(); i++)cout << a[i] << " ";
	cout << endl;
}

int main() {
	std::vector<int> a = { 1,2,3,4,5 };
	print(a);
	cout << a.size() << " " << a.empty();
}
```

### begin()/end()

- `begin()`: 返回起始位置的迭代器（左闭）；

- `end()`: 返回终点位置的下一个位置的迭代器（右开）。

利用迭代器可以访问整个`vector`，存在迭代器的容器就可以使用范围 for 遍历。

::: code-group

```c++ [写法1]
// 利用迭代器来遍历
void print(vector<int>& a) {
	for (vector<int>::iterator it = a.begin(); it != a.end(); it++)cout << *it << " ";
	cout << endl;
}
```

```c++ [写法2]
// 利用范围 for 来遍历
void print(vector<int>& a) {
	for (auto x : a)cout << x << " ";
	cout << endl;
}
```

:::

### push_back()/pop_back()

- `push_back()`: 在顺序表尾部添加一个元素；

- `pop_back()`: 删除顺序表尾部的一个元素。

> 当然还有`insert()`和`erase()`。不过由于事件复杂度过高，尽量不使用。

```c++
#include<iostream>
#include<vector>
using namespace std;

void print(vector<int>& a) {
	for (auto x : a)cout << x << " ";
	cout << endl;
}

int main() {
	std::vector<int> a = { 1,2,3,4,5 };
	print(a);
	a.pop_back();
	print(a);
	a.push_back(6);
	print(a);
}
```

### front()/back()

- `front()`: 返回顺序表第一个元素；

- `back()`: 返回顺序表最后一个元素。

```c++
#include<iostream>
#include<vector>
using namespace std;

int main() {
	std::vector<int> a = { 1,2,3,4,5 };
	cout << a.front() << " " << a.back();
}
```

### resize()

- `resize()`: 修改顺序表的大小。

- 如果大于原始的大小，多出来的位置会补上默认值，一般是 0。

- 如果小于原始的大小，相当于把后面的元素全部删掉。

```c++
#include<iostream>
#include<vector>
using namespace std;

void print(vector<int>& a) {
	for (auto x : a)cout << x << " ";
	cout << endl;
}

int main() {
	std::vector<int> a(4,4);
	print(a);
	a.resize(3);
	print(a);
	a.resize(6);
	print(a);
}
```

### clear()

- `clear()`: 清空顺序表。

```c++
#include<iostream>
#include<vector>
using namespace std;

void print(vector<int>& a) {
	for (auto x : a)cout << x << " ";
	cout << endl;
}

int main() {
	std::vector<int> a(4,4);
	print(a);
	a.clear();
	print(a);
}
```

## 双向链表-list

## 栈-stack

### 创建 stack

```c++
#include<stack>

int main() {
	std::stack<int> a;
}
```

`T`可以是任意类型的数据。

### size()/empty()

- `size()`: 返回栈里实际元素的个数；

- `empty()`: 栈是否为空。如果为空：返回 true，否则返回 false。

### push()/pop()

- `push()`: 往栈里添加一个元素；

- `pop()`: 删除栈顶的一个元素。

### top()

- `top()`: 返回栈顶元素，但是不会删除栈顶元素。

```c++
#include<iostream>
#include<stack>
using namespace std;

int main() {
	stack<int> st;
	for (int i = 1; i <= 10; i++)st.push(i);
	while (st.size()) {
		cout << st.top() << endl;
		st.pop();
	}
}
```
