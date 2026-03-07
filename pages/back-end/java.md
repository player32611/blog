# Java

> Java 语言是一个面向对象的程序设计语言，除了面向对象的特点意外， Java 语言还在安全性、平台无关性、支持多线程、内存管理等许多方面具有卓越的优点

::: danger 警告
该页面尚未完工!
:::

::: details 目录

[[toc]]

:::

## JDK

Java 的产品加 **JDK**（Java Development Kit：Java 开发工具包），必须安装 JDK 才能使用 Java。

下载链接：https://www.oracle.com/cn/java/

### JDK 的组成

- JVM（Java Virtual Machine）：Java 虚拟机，真正运行 Java 程序的地方。

- 核心类库：Java 自己写好的程序，给程序员自己的程序调用的。

- JRE（Java Runtime Environment）：Java 的运行环境

- JDK（Java Development Kit）：Java 开发工具包（包括上面所有）

::: details Java 与 C++ 的区别

Java 中没有 `#include` 和 `#define` 等预处理功能，用 `import` 语句包含其它类和包

Java 中没有 `structure`，`union` 及 `typedef`

Java 中没有不属于类成员的函数，没有指针和多重继承，Java 只支持单重继承

Java 中禁用 `goto`，但 `goto` 还是保留的关键字

Java 中没有操作符重载

Java 中没有全局变量，可以在类中定义公用、静态的数据成员实现相同功能

:::

## IDEA

### 使用 IDEA 开发 Java 程序的步骤

- project -> module -> package -> class

- 一个 project 中可以创建多个 module

- 一个 module 中可以创建多个 package

- 一个 package 中可以创建多个 class

IDEA 中的 java 程序是自动编译和执行的，编译后的 class 问价在工程路径下的一个 out 文件夹里。

### 常用快捷键

- `main/psvm`、 `sout` 等：快速键入相关代码

- `ctrl + alt + l`：格式化代码

## 基础语法

### 特殊字符

| 特殊字符 |   描述   |
| :------: | :------: |
|    \n    |   换行   |
|    \t    | Tab 缩进 |

### 标识符

**标识符**是代码中所有我们自己起的名字。

::: tip 标识符的命名规则-硬性要求

- 由数组、字母、下划线、美元符 `$` 组成

- 不能以数字开头

- 不能是关键字

- 区分大小写

:::

::: tip 标识符的命名规则-软性建议

- **小驼峰命名法**：用于方法，变量。一个单词时全部小写，多个单词时第一个字母全部小写，其它单词首字母大写

- **大驼峰命名法**：用于类，接口。一个单词时首字母大写，多个单词时每个单词首字母大写

:::

### 逻辑运算符

| 逻辑运算符 |    描述    |
| :--------: | :--------: |
|    `&`     | 与（而且） |
|    `\|`    | 或（或者） |
|    `!`     | 非（取反） |

::: tip 短路逻辑运算符

**运行规则**：和单个的 `&`、`|` 是一样的，只不过提高了效率

- `&&`：**短路与**，当第一个条件为 `true` 时，才会判断第二个条件，当第一个条件为 `false` 时，第二个条件不会判断

- `||`：**短路或**，当第一个条件为 `true` 时，返回 `true`，当第一个条件为 `false` 时，才会判断第二个条件

:::

### 键盘录入

**键盘录入**：获取键盘按下的数据，并保存在变量当中

1. 导入 `java.util.Scanner` 类：

```java
import java.util.Scanner;
```

2. 创建 Scanner 对象：

```java
Scanner sc = new Scanner(System.in);
```

3. 获取数据：

```java
int a = sc.nextInt();
String b = sc.next();
double c = sc.nextDouble();
```

### 输出

**输出**：将数据输出到控制台

```java
System.out.println("输出内容");
```

### 类型转换

**类型转换**：将一种数据类型转换成另一种数据类型

- 隐式转换：不同类型的数据进行计算时，默认采用隐式转换，Java 自动转换，把取值范围小的提升为取值范围大的，再进行计算（如有 byte short 类型的数据，先提升为 int 类型）

- 强制转换：不会主动触发，需要手动书写代码，有可能导致精度丢失

```java
int a = 10;
byte b = (byte) a;
```

### 字符串运算

字符串只有 `+` 操作，没有其它操作

任意数据 + 字符串都是拼接操作，并产生一个新的字符串

```java
String a = 123 + "hello";
// a = "123hello"
String b = 10 + 8 + "岁"
// b = "18岁";
String c = 10 + 8 + "岁" + 1 + 2;
// c = "18岁12"
```

### switch 语句新特性

在 JDK14 及以后的版本中，switch 语句可以采用另一种方式进行书写：

```java
switch (expression) {
  case value1 ->{
    statement1;
  }
  case value2 ->{
    statement2;
  }
  default ->{
    statement3;
  }
}
```

```java
switch (expression) {
  case value1 -> statement1;
  case value2 -> statement2;
  default -> statement3;
}
```

这种情况下，不会发生 case 穿透，即不会执行后面的 case。

如何想要穿透，可以采用以下写法：

```java
int result = switch (expression) {
  case value1 -> {
    yield statement1;
  }
  case value2 -> {
    yield statement2;
  }
  default -> {
    yield statement3;
  }
};
```

```java
int result = switch (expression) {
  case value1 -> statement1;
  case value2 -> statement2;
  default -> statement3;
};
```

这种写法表示将语句结果返回给 `result`。

## 数组

::: danger 警告

该部分尚未完工!

:::

### 数组的声明

```java
Type[] arrayName;
Type arrayName[];
```

### 数组的创建

```java
arrayName = new Type[components number];
```
