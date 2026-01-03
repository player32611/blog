# 基本语法

::: danger 警告
该页面尚未完工!
:::

## 标题

**输入**

```YAML
# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题
```

**输出**

# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题

## 表格

**输入**

```YAML
| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |
```

**输出**

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

## 框

**输出框-YAML**

`````YAML
    ````YAML
    //内容
    ````
`````

**输出框-md**

`````md
    ````md
    //内容
    ````
`````

**输出框-js**

````js
```js
    //内容 
    ```;
````

**信息框**

::: info 信息
::: info
这是信息框
:::
:::

::: tip 提示
::: tip
这是提示框
:::
:::

::: warning 警告
::: warning
这是警告框
:::
:::

::: danger 危险
::: danger
这是危险警告框
:::
:::

::: details 详情
::: details
这是详情框
:::
:::

::: info
::: tip
这是框中框
:::

## 字体

**输入**

```YAML
正常字体

*倾斜字体*

**加粗字体**

***加粗倾斜字体***
```

**输出**

正常字体

_倾斜字体_

**加粗字体**

**_加粗倾斜字体_**

## 更多信息

更多语法请点击[此处](https://vitepress.dev/guide/markdown).
