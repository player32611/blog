# Node.js

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## 初识 Node.js

**Node.js** 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。

::: warning 注意

- 浏览器是 JavaScript 的前端运行环境

- Node.js 是 JavaScript 的后端运行环境

- Node.js 中无法调用 DOM 和 BOM 等浏览器内置 API

:::

浏览器中的 JavaScript 学习路径：JavaScript 基础语法 + 浏览器内置 API（DOM + BOM） + 第三方库（jQuery、art-template）

Node.js 的学习路径：JavaScript 语言基础 + Node.js 内置 API 模块（fs、path、http） + 第三方 API 模块（express、mysql）

## fs 文件系统模块

**fs 模块**是 Node.js 官方提供的、用来操作文件的模块。它提供了一系列的方法的属性，用来满足用户对文件的操作需求。

如果要在 JavaScript 代码中，使用 fs 模块来操作文件，则需要使用如下的方式先导入它：

```javascript
const fs = require("fs");
```

```javascript
import fs from "fs";
```

### 读取指定文件中的内容

使用 `fs.readFile()` 方法，可以读取指定文件中的内容，语法格式如下：

```javascript
fs.readFile(path[, options], callback)
```

- **path**：必选参数，字符串，表示文件的路径

- **options**：可选参数，表示以什么编码格式来读取文件

- **callback**：必选参数，文件读取完成后，通过回调函数拿到读取的结果

例如以 utf-8 的编码格式，读取指定文件的内容，并打印 `err` 和 `dataStr` 的值：

```javascript
const fs = require("fs");
fs.readFile("./files/1.txt", "utf-8", function (err, dataStr) {
  console.log(err);
  console.log("-----");
  console.log(dataStr);
});
```

::: tip 提示

如果读取成功，则 `err` 的值为 `null`

如果读取失败，则 `dataStr` 的值为 `undefined`

:::

因此，可以判断 `err` 对象是否为 `null`，从而知晓文件读取的结果：

```javascript
fs.readFile("./README.md", "utf-8", function (err, dataStr) {
  if (err) {
    return console.log("读取文件失败！" + err.message);
  }
  console.log("读取文件成功！" + dataStr);
});
```

### 向指定的文件中写入内容

使用 `fs.writeFile()` 方法，可以向指定的文件中写入内容，语法格式如下：

```javascript
fs.writeFile(file, data[, options], callback)
```

- **file**：必选参数，需要指定一个文件路径的字符串，表示文件的存放路径

- **data**：必选参数，表示要写入的内容

- **options**：可选参数，表示以什么格式写入文件内容，默认值是 utf-8

- **callback**：必选参数，文件写入完成后的回调函数

```javascript
fs.writeFile("./files/1.txt", "Hello Node.js", function (err) {});
```

::: warning 注意

- `fs.writeFile()` 方法只能用来创建文件，不能用来创建路径

- 重复调用 `fs.writeFile()` 写入同一个文件，新写入的内容会覆盖之前的旧内容

:::

::: tip 提示

如果文件写入成功，则 `err` 的值为 `null`

如果文件写入失败，则 `err` 的值为一个错误对象

:::

因此，可以判断 `err` 对象是否为 `null`，从而知晓文件写入的结果：

```javascript
fs.writeFile("./pages/back-end/nodejs.md", "Hello node js", function (err) {
  if (err) {
    return console.log("写入文件失败！" + err.message);
  }
  console.log("写入文件成功！");
});
```

::: details 具体示例

使用 fs 文件系统模块，将文件进行整理：

整理前文件中的数据格式如下：

```
小红=99 小白=100 小黄=70 小黑=66 小绿=88
```

整理后文件中的数据格式如下：

```
小红：99
小白：100
小黄：70
小黑：66
小绿：88
```

```javascript
// 导入需要的 fs 文件系统模块
const fs = require("fs");

// 调用 fs.readFile() 读取文件的内容
fs.readFile("./files/1.txt", "utf-8", function (err, dataStr) {
  // 判断是否读取成功
  if (err) {
    return console.log("读取文件失败！" + err.message);
  }

  // 先把数据按照空格进行分割
  const arrOld = dataStr.split(" ");

  // 循环分割后的数据，对每一项数据，进行字符串的替换操作
  const arrNew = [];
  arrOld.forEach((item) => {
    arrNew.push(item.replace("=", "："));
  });

  // 把新数组中的每一项，进行合并，得到一个新的字符串
  const newStr = arrNew.join("\r\n");

  // 调用 fs.writeFile() 方法，将处理结果，写入到文件中
  fs.writeFile("./files/1.txt", newStr, function (err) {
    if (err) {
      return console.log("写入文件失败！" + err.message);
    }
    console.log("写入文件成功！");
  });
});
```

:::

### 路径动态拼接

在使用 fs 模块操作文件时，如果提供的操作路径是以 `./` 或 `../` 开头的相对路径时，很容易出现路径动态拼接错误的问题。

原因是代码在运行的时候，会以**执行 node 命令时所处的目录**，动态拼接出被操作文件的完整路径。

解决方案为在使用 fs 模块操作文件时，直接提供完整的路径，不要提供 `./` 或 `../` 开头的相对路径，从而防止路径动态拼接的问题。

或者使用 `__dirname` 获取当前文件所处的目录，再拼接出被操作文件的完整路径。

```javascript
fs.readFile(__dirname + "/files/1.txt", "utf-8", function (err, dataStr) {
  if (err) {
    return console.log("读取文件失败！" + err.message);
  }
  console.log("读取文件成功！" + dataStr);
});
```

## path 路径模块

**path 模块**是 Node.js 官方提供的，用来处理路径的模块。它提供了一系列的方法和属性，用来满足用户对路径的处理需求。

如果要在 JavaScript 代码中，使用 path 模块来处理路径，则需要使用如下的方式先导入它：

```javascript
const path = require("path");
```

```javascript
import path from "path";
```

### 路径拼接

使用 `path.join()` 方法，可以把多个路径片段拼接为完整的路径字符串，语法格式如下：

```javascript
path.join([...paths]);
```

- **...paths**: 路径片段的序列

- **返回值**：string 类型

::: details 具体示例

```javascript
const pathStr = path.join("/a", "/b/c", "../", "./d", "e");
console.log(pathStr); // 输出 /a/b/d/e

const pathStr2 = path.join(__dirname, "./files/1.txt");
console.log(pathStr2); // 输出当前文件所处目录 \files\1.txt
```

:::

### 获取路径中的文件名

使用 `path.basename()` 方法，可以获取路径中的最后一部分，经常通过这个方法获取路径中的文件名，语法格式如下：

```javascript
path.basename(path[, ext])
```

- **path**: 必选参数，表示一个路径的字符串

- **ext**: 可选参数，文件扩展名

- **返回值**: 表示路径中的最后一部分

::: details 具体示例

```javascript
const fpath = "/a/b/c/index.html"; // 文件的存放路径

const fullName = path.basename(fpath);
console.log(fullName); // 输出 index.html

const nameWithoutExt = path.basename(fpath, ".html");
console.log(nameWithoutExt); // 输出 index
```

:::

### 获取路径中的文件扩展名

使用 `path.extname()` 方法，可以获取路径中的扩展名部分，语法格式如下：

```javascript
path.extname(path);
```

- **path**: 表示一个路径的字符串

- **返回值**: 返回得到的扩展名字符串

::: details 具体示例

```javascript
const fpath = "/a/b/c/index.html"; // 路径字符串

const fext = path.extname(fpath);
console.log(fext); // 输出 .html
```

:::

::: details 综合案例 - 拆分出 html、css、js 文件

**案例的实现步骤**：

① 创建两个正则表达式，分别用来匹配 `<style>` 和 `<script>` 标签

② 使用 fs 模块，读取需要被处理的 HTML 文件

③ 自定义 `resolveCSS` 方法，来写入 index.css 样式文件

④ 自定义 `resolveJS` 方法，来写入 index.js 脚本文件

⑤ 自定义 `resolveHTML` 方法，来写入 index.html 文件

```javascript
const regStyle = /<style>[\s\S]*<\/style>/; // 匹配 style 标签的正则
const regScript = /<script>[\s\S]*<\/script>/; // 匹配 script 标签的正则

// 自定义 resolveCSS 方法
const resolveCSS = (htmlStr) => {
  // 1. 使用正则来匹配 <style>...</style>
  const r1 = regStyle.exec(htmlStr);
  // 2. 如果匹配成功，就调用 fs.writeFile 将结果写入到 css 文件中
  const newCSS = r1[0].replace("<style>", "").replace("</style>", "");
  fs.writeFile(
    path.join(__dirname, "./clock/index.css"),
    newCSS,
    function (err) {
      if (err) return console.log("写入 CSS 失败！" + err.message);
      console.log("写入 CSS 样式文件成功！");
    },
  );
};

// 自定义 resolveJS 方法
const resolveJS = (htmlStr) => {
  // 1. 使用正则来匹配 <script>...</script>
  const r2 = regScript.exec(htmlStr);
  // 2. 如果匹配成功，就调用 fs.writeFile 将结果写入到 js 文件中
  const newJS = r2[0].replace("<script>", "").replace("</script>", "");
  fs.writeFile(path.join(__dirname, "./clock/index.js"), newJS, function (err) {
    if (err) return console.log("写入 JavaScript 脚本失败！" + err.message);
    console.log("写入 JavaScript 脚本成功！");
  });
};

// 自定义 resolveHTML 方法
const resolveHTML = (htmlStr) => {
  const newHTML = htmlStr
    .replace(regStyle, '<link rel="stylesheet" href="./index.css"/>')
    .replace(regScript, '<script src="./index.js"></script>');
  fs.writeFile(path.join(__dirname, "./clock/index.html"), newHTML, (err) => {
    if (err) return console.log("写入 HTML 文件失败！" + err.message);
    console.log("写入 HTML 文件成功！");
  });
};

// 读取需要被处理的 HTML 文件
fs.readFile(path.join(__dirname, "./index.html"), "utf-8", (err, dataStr) => {
  if (err) {
    return console.log("读取 HTML 文件失败！" + err.message);
  }
  resolveCSS(dataStr);
  resolveJS(dataStr);
  resolveHTML(dataStr);
});
```

:::

## http 模块

::: danger 警告

该部分尚未完工!

:::
