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

**http 模块**是 Node.js 官方提供的，用于创建 web 服务器的模块。通过 http 模块，就能方便的把一台普通的电脑，变成一台 Web 服务器，从而对外提供 Web 资源服务。

如果要希望使用 http 模块创建 Web 服务器，则需要先导入它：

```javascript
const http = require("http");
```

```javascript
import http from "http";
```

::: details 进一步理解 http 模块的作用

服务器和普通电脑的区别在于，服务器上安装了 web 服务器软件，例如：IIS、Apache 等。通过安装这些服务器软件，就能把一台普通的电脑变成一台 web 服务器。

在 Node.js 中，我们不需要使用 ISS、Apache 等这些第三方 web 服务器软件。因为我们可以基于 Node.js 提供的 http 模块，通过几行简单的代码，就能轻松的手写一个服务器软件，从而对外提供 web 服务。

:::

### IP 地址

**IP 地址**就是互联网上每台计算机的唯一地址，因此 IP 地址具有唯一性。

IP 地址的格式通常用 "点分十进制" 表示成（a.b.c.d）的形式，其中，a、b、c、d 都是 0~255 之间的十进制整数。

::: warning 注意

互联网中每台 Web 服务器，都有自己的 IP 地址

在开发期间，自己的电脑即是一台服务器，也是一个客户端，为了方便测试，可以在自己的浏览器中输入 127.0.0.1 这个 IP 地址，就能把自己的电脑当作一台服务器进行访问了

:::

### 域名和域名服务器

尽管 IP 地址能够唯一地标记网络上的计算机，但 IP 地址是一长串数子，不直观，而且不便于记忆，于是人们又发明了另一套字符型的地址方案，即所谓的**域名**（Domain Name）地址。

IP 地址和域名是相对应的关系，这份对应关系存放在一种叫做**域名服务器**（DNS，Domain Name Server）的电脑中。使用者只需通过好记的域名访问对应的服务器即可，对应的转换工作由域名服务器实现。因此，域名服务器就是提供 IP 地址和域名之间的转换服务的服务器。

::: warning 注意

单纯使用 IP 地址，互联网中的电脑也能够正常工作。但是有了域名的加持，能让互联网的世界变得更加方便

在开发测试期间，127.0.0.1 对应的域名是 localhost，它们都代表我们自己的这台电脑，在使用效果上没有任何区别

:::

### 端口号

计算机中的端口号，就好像是显示生活中的门牌号一样。通过门牌号，外卖小哥可以在整栋大楼众多的房间中，准确把外卖送到你的手中。

同样的道理，在一台电脑中，可以运行成百上千个 web 服务。每个 web 服务都对应一个唯一的端口号。客户端发送过来的网络请求，通过端口号，可以准确地交给对应的 web 服务进行处理。

::: warning 注意

每个端口号不能被多个 web 服务占用

在实际应用中，URL 中的 80 端口可以被省略

:::

### 创建最基本的 web 服务器

**创建 web 服务器的基本步骤**：

① 导入 http 模块

② 创建 web 服务器实例

③ 为服务器实例绑定 request 事件，监听客户端的请求

④ 启动服务器

```javascript
import http from "http";

// 创建 web 服务器实例
const server = http.createServer();

// 使用服务器实例的 .on() 方法，即可监听客户端发送过来的网络请求
server.on("request", (req, res) => {
  // 只要有客户端来请求我们自己的服务器，就会触发 request 事件，从而调用这个事件处理函数
  console.log("Some visit our web server.");
});

// 调用 server.listen(端口号, 回调函数) 方法，即可启动 web 服务器
server.listen(80, () => {
  console.log("http server running at http://127.0.0.1");
});
```

### req 请求对象

只要服务器接收到了客户端的请求，就会调用通过 `server.on()` 为服务器绑定的 request 事件处理函数。

如果想在事件处理函数中，访问与客户端相关的数据或属性，可以使用如下的方式：

```javascript
server.on("request", (req, res) => {
  // req.url 是客户端请求的 URL 地址
  const url = req.url;
  // req.method 是客户端请求的 method 类型
  const method = req.method;
  const str = `Your request url is ${url}, and request method is ${method}`;
  console.log(str);
});
```

::: tip 提示

req 是请求对象，包含了与客户端相关的数据和属性。

:::

### res 响应对象

在服务器的 request 事件处理函数中，如果想访问与服务器相关的数据活属性，可以使用如下的方式：

```javascript
server.on("request", (req, res) => {
  const url = req.url;
  const method = req.method;
  const str = `Your request url is ${url}, and request method is ${method}`;
  console.log(str);
  // 调用 res.end() 方法，向客户端响应一些内容
  res.end(str);
});
```

### 解决中文乱码问题

当调用 res.end() 方法，向客户端发送中文内容的时候，会出现乱码问题，此时，需要手动设置内容的编码格式：

```javascript
server.on("request", (req, res) => {
  // 发送的内容包含中文
  const str = `您请求的 url 地址是 ${req.url}, 请求的 method 类型是 ${req.method}`;
  // 为了防止中文显示乱码的问题，需要设置响应头 Content-Type 为 text/html；charset=utf-8
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(str);
});
```

### 根据不同的 url 响应不同的 html 内容

**核心实现步骤**：

① 获取请求的 url 地址

② 设置默认的响应内容为 404 Not Found

③ 根据不同的 url 地址，设置不同的响应内容

④ 设置 Content-Type 响应头，防止中文乱码

⑤ 使用 res.end() 把内容响应给客户端

```javascript
server.on("request", (req, res) => {
  const url = req.url; // 获取请求的 url 地址
  let content = "<h1>404 Not Found</h1>"; // 设置默认的内容为 404 Not Found
  if (url === "/" || url === "/index.html") {
    content = "<h1>首页</h1>"; // 用户请求的是首页
  } else if (url === "/about.html") {
    content = "<h1>关于</h1>"; // 用户请求的是关于页面
  }
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(content);
});
```

::: details 综合案例 - 通过服务器访问网页文件

**核心思路**：把文件的实际存放路径，作为每个资源的请求 url 地址

**实现步骤**：

① 导入需要的模块

② 创建基本的 web 服务器

③ 将资源的请求 url 地址映射为文件的存放路径

④ 读取文件内容并响应给客户端

```javascript
// 1 导入需要的模块
const http = require("http");
const path = require("path");
const fs = require("fs");

// 2.1 创建 web 服务器
const server = http.createServer();

// 2.2 监听 web 服务器的 request 事件
server.on("request", (req, res) => {
  // 3.1 获取到客户端请求的 url 地址
  const url = req.url; // 获取请求的 url 地址
  // 3.2 把请求的 url 地址映射为本地文件的存放路径
  const fpath = path.join(__dirname, url);
  // 4.1 根据映射过来的文件路径读取文件
  fs.readFile(fpath, "utf8", (err, dataStr) => {
    // 4.2 读取文件失败后，向客户端响应固定的 "错误消息"
    if (err) return res.end("404 Not Found");
    // 4.3 读取文件成功后，将 "读取成功的内容" 响应给客户端
    res.end(dataStr);
  });
});

// 2.3 启动 web 服务器
server.listen(80, () => {
  console.log("http server running at http://127.0.0.1");
});
```

⑤ 优化资源的请求路径

```javascript
server.on("request", (req, res) => {
  const url = req.url; // 获取请求的 url 地址
  // 5.1 预定义空白的文件存放路径
  let fpath = "";
  if (url === "/") {
    // 5.2 如果请求的路径是否为 /，则手动指定文件的存放路径
    fpath = path.join(__dirname, "./clock/index.html");
  } else {
    fpath = path.join(__dirname, "clock", url);
  }
  fs.readFile(fpath, "utf8", (err, dataStr) => {
    // 4.2 读取文件失败后，向客户端响应固定的 "错误消息"
    if (err) return res.end("404 Not Found");
    // 4.3 读取文件成功后，将 "读取成功的内容" 响应给客户端
    res.end(dataStr);
  });
});
```

:::

## 模块化

**模块化**是指解决一个复杂问题时，自顶层向下逐层把系统划分成若干模块的过程。对于整个系统来说，模块是可组合、分解和更换的单元。

在编程邻域中的模块化，就是遵守固定的规则，把一个大文件拆分成独立并互相依赖的多个小模块。

::: tip 把代码进行模块化拆分的好处

- 提高了代码的复用性

- 提高了代码的可维护性

- 可以实现按需加载

:::

**模块化规范**就是对代码进行模块化的拆分与组合时，需要遵守的那些规则。

例如：

- 使用什么样的语法格式来引用模块

- 在模块中使用什么样的语法格式向外暴露成员

### Node.js 中的模块化

Node.js 中根据模块化来源的不同，将模块分为了 3 大类，分别是：

- **内置模块**：由 Node.js 官方提供的，例如 fs、path、http 等

- **自定义模块**：用户创建的每个 .js 文件，都是自定义模块

- **第三方模块**：由第三方开发出来的模块，并非官方提供的内置模块，也不是用户创建的自定义模块，使用前需要先下载

### Node.js 中的模块作用域

和函数作用域类似，在自定义模块中定义的变量、方法等成员，只能在当前模块内被访问，这种模块级别的访问限制，叫做**模块作用域**。

::: tip 模块作用域的好处

防止了全局变量污染的问题

:::

### 模块的加载机制

模块在第一次加载后会被缓存，这也意味着多次调用 `require()` 不会导致模块的代码被执行多次。

::: warning 注意

不论是内置模块、用户自定义模块、还是第三方模块，它们都会优先从缓存中加载，从而提高模块的加载效率

:::

内置模块的加载优先级最高，即使在 node_modules 目录下有名字相同的模块，始终返回内置的模块。

加载自定义模块时，必须指定以 ./ 或 ../ 开头的路径标识符。如果没有指定，则 node 会把它当作内置模块或第三方模块进行加载。

同时如果省略的文件的扩展名，则 Node.js 会按顺序分别尝试加载以下的文件：

① 按照确切的文件名进行加载

② 补全 .js 扩展名进行加载

③ 补全 .json 扩展名进行加载

④ 补全 .node 扩展名进行加载

⑤ 加载失败，终端报错

如果传递给 `require()` 的模块标识符不是一个内置模块，也没有以 ./ 或 ../ 开头，则 Node.js 会从当前模块的父目录开始，尝试从 /node_modules 文件夹中加载第三方模块。

如果没有找到对应的第三方模块，则移动到再上一层父目录中，进行加载，直到文件系统的根目录。

### 目录作为模块

当把目录作为模块标识符，传递给 `require()` 进行加载的时候，有三种加载方式：

① 在被加载的目录下查找一个叫做 package.json 的文件，并寻找 main 属性，作为 `require()` 加载的入口

② 如果目录里没有 package.json 文件，或者 main 入口不存在或无法解析，则 Node.js 将会试图加载目录下的 index.js 文件

③ 如果以上两步都失败了，则 Node.js 会在终端打印错误消息，报告模块的缺失：`Error: Cannot find module 'xxx'`

### module 对象

在每个 .js 自定义模块中都有一个 module 对象，它里面存储了和当前模块有关的信息，打印如下：

```javascript
console.log(module);
```

```console
{
  id: '.',
  path: 'e:\\files\\我的git库\\blog',
  exports: {},
  filename: 'e:\\files\\我的git库\\blog\\test.js',
  loaded: false,
  children: [],
  paths: [
    'e:\\files\\我的git库\\blog\\node_modules',
    'e:\\files\\我的git库\\node_modules',
    'e:\\files\\node_modules',
    'e:\\node_modules'
  ],
  Symbol(kIsMainSymbol): true,
  Symbol(kIsCachedByESMLoader): false,
  Symbol(kURL): undefined,
  Symbol(kFormat): undefined,
  Symbol(kIsExecuting): true
}
```

### CommonJS 模块化规范

CommonJS 规定：

- 每个模块内部，`module` 变量代表当前模块

- `module` 变量是一个对象，它的 `exports` 属性（即 `module.exports`）是对外的接口

- 加载某个模块，其实是加载该模块的 `module.exports` 属性。`require()` 方法用于加载模块

## Express

Express 是基于 Node.js 平台，快速、开放、极简的 Web 开发框架。

Express 的作用和 Node.js 内置的 http 模块类似，是专门用来创建 Web 服务器的。

**Express 的本质**：就是一个 npm 上的第三方包，提供了快速创建 Web 服务器的便捷方法。

对于前端程序员来说，最常见的两种服务器，分别是：

- **Web 网站服务器**：专门对外提供 Web 网页资源的服务器

- **API 接口服务器**：专门对外提供 API 接口的服务器

使用 Express，我们可以方便、快速的创建 Web 网站的服务器或 API 接口的服务器。

在项目所处的目录中，运行如下的终端命令，即可将 express 安装到项目中使用：

```bash
npm install express
```

### 创建基本的 Web 服务器

```javascript
import express from "express";

// 创建 web 服务器
const app = express();

// 调用 app.listen(端口号, 回调函数)，启动服务器
app.listen(80, () => {
  console.log("express server running at http://127.0.0.1");
});
```

通过 `app.get()` 方法，可以监听客户端的 GET 请求，具体的语法格式如下：

```javascript
app.get(url, (req, res) => {});
```

- **参数 1**：客户端请求的 URL 地址

- **参数 2**：请求对应的处理函数
  - **req**：请求对象（包含了与请求相关的属性与方法）
  - **res**：响应对象（包含了与响应相关的属性与方法）

通过 `res.send()` 方法，可以把处理好的内容，发送给客户端：

```javascript
res.send({});
```

::: details 具体示例

监听客户端的 GET 和 POST 请求，并向客户端响应对应的内容：

```javascript
import express from "express";

const app = express();

// 监听客户端的 GET 和 POST 请求，并向客户端响应对应的内容
app.get("/user", (req, res) => {
  // 调用 express 提供的 res.send() 方法，向客户端响应一个 JSON 对象
  res.send({ name: "zs", age: 20, gender: "男" });
});

app.post("/user", (req, res) => {
  // 调用 express 提供的 res.send() 方法，向客户端响应一个文本字符串
  res.send("请求成功");
});

app.listen(80, () => {
  console.log("express server running at http://127.0.0.1");
});
```

:::

通过 `req.query` 对象，可以访问到客户端通过查询字符串的形式（`?name=za&age=20`），发送到服务器的参数：

```javascript
app.get("/", (req, res) => {
  console.log(req.query);
  res.send({ name: req.query.name, age: req.query.age, gender: "男" });
});
```

默认情况下，`req.query` 是一个空对象。

通过 `req.params` 对象，可以访问到 URL 中，通过 `:` 匹配到的动态参数：

```javascript
app.get("/user/:id", (req, res) => {
  console.log(req.params);
  res.send(req.params);
});
```

默认情况下，`req.params` 是一个空对象。

通过 `express.static()`，我们可以非常方便地创建一个静态资源服务器。

例如，通过以下代码就可以将 public 目录下的图片、CSS 文件、JavaScript 文件对外开放访问了：

```javascript
app.use(express.static("public"));
```

此时即可通过 URL 路径直接访问对应文件。

::: warning 注意

Express 在指定的静态目录中查找文件，并对外提供资源的访问路径。

因此，存放静态文件的目录名不会出现在 URL 中。

:::

如果要托管多个静态资源目录，请多次调用 `express.static()` 函数：

```javascript
app.use(express.static("public"));
app.use(express.static("./files"));
```

::: warning 注意

访问静态资源文件时，`express.static()` 函数会根据目录的添加顺序查找所需的文件

:::

如果希望在托管的静态资源访问路径之前，挂载路径前缀（即在访问的 URL 中添加静态资源目录），则可以使用如下的方式：

```javascript
app.use("/abc", express.static("./files"));
```

### Express 路由

广义上来讲，路由就是映射关系。

在 Express 中，路由指的是客户端的请求与服务器处理函数之间的映射关系。

Express 中的路由分 3 部分组成，分别是**请求的类型**、**请求的 URL 地址**、**处理函数**。

::: details 具体示例

```javascript
// 匹配 GET 请求，且请求 URL 为 /
app.get("/", (req, res) => {
  res.send("Hello World");
});

// 匹配 POST 请求，且请求 URL 为 /
app.post("/", (req, res) => {
  res.send("Got a POST request");
});
```

:::

每当一个请求到达服务器之后，需要先经过路由的匹配，只有匹配成功之后，才会调用对应的处理函数。

在匹配时，会按照路由的顺序进行匹配，如果请求类型和请求的 URL 同时匹配成功，则 Express 会将这次请求，转交给对应的 function 函数进行处理。

::: warning 注意

- 按照定义的先后顺序进行匹配

- 请求类型和请求的 URL 同时匹配成功，才会调用对应的处理函数

:::

在 Express 中使用路由最简单的方式，就是把路由挂载到 app 上，示例如下：

```javascript
import express from "express";

const app = express();

// 挂载路由
app.get("/", (req, res) => {
  res.send("Hello World.");
});
app.post("/", (req, res) => {
  res.send("Hello World.");
});

// 启动 Web 服务器
app.listen(80, () => {
  console.log("express server running at http://127.0.0.1");
});
```

### 模块化路由

为了方便对路由进行模块化的管理，Express 不建议将路由直接挂载到 app 上，而是推荐将路由抽离为单独的模块。

将路由抽离为单独模块的步骤如下：

① 创建路由模块对应的 .js 文件

② 调用 `express.Router()` 函数创建路由对象

③ 向路由对象上挂载具体的路由

④ 向外共享路由对象

⑤ 使用 `app.use()` 函数注册路由模块

::: code-group

```javascript [router.js]
import express from "express";

// 创建路由对象
const router = express.Router();

router.get("/user/list", (req, res) => {
  res.send("Get user list");
});

router.post("/user/add", (req, res) => {
  res.send("Add new user");
});

export default router;
```

```javascript [index.js]
import express from "express";
// 导入路由模块
import router from "./router/router.js";

const app = express();

// 注册路由模块
app.use(router);

app.listen(80, () => {
  console.log("express server running at http://127.0.0.1");
});
```

:::

::: tip `app.use()`

`app.use()` 函数的作用，就是来注册全局中间件

:::

类似于托管静态资源时，为静态资源统一挂载访问前缀一样，路由模块添加前缀的方式也非常简单：

```javascript
app.use("/api", router);
```

### Express 中间件

::: danger 警告

该部分尚未完工!

:::
