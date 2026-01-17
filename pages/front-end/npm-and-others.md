# npm与其他东西

::: danger 警告
该页面尚未完工!
:::

::: details 目录

[[toc]]

:::

## npm 的使用

`npm -v` 通过查看版本，看 npm 是否安装成功

`npm install <Module Name>` 使用 npm 命令安装模块d

`npm install <Module Name> -g ` 全局安装

`npm list -g` 查看所有全局安装的模块

`npm list vue` 查看某个模块的版本号

`npm -g install npm@5.9.1` 安装指定版本的 npm

`npm install -save moduleName`

`npm init` 生成 package.json 配置文件

## package.json 属性说明

`name` 包名

`version` 包的版本号

::: tip 提示

package.json 文件中版本号的说明，安装的时候代表不同的含义:

"5.03" 表示安装指定的5.0.5版本

"~5.0.3" 表示安装5.0.X中最新的版本

"^5.0.5" 表示安装5.X.X中最新的版本

:::

`description` 包的描述

`homepage` 包的官网 url

`author` 包的作者姓名

`contributors` 包的其他贡献者姓名

`dependencies` 依赖包列表。如果依赖包没有安装， npm 会自动将依赖包安装在 node_module 目录下。

`devDependencies` 开发时依赖包列表。

`repository` 包代码存放的地方的类型，可以是 git 或 svn ，git 可在 Github 上

`main` 指定了程序的主入口文件， require('moduleName') 就会加载这个文件。这个字段的默认值是模块根目录下面的 index.js 。

`keywords` 关键字

## 解决 ES6 语法兼容性问题

1. 使用 npm 全局安装 babel-cli 包 `npm install babel-cli -g`

2. 在项目目录下新建 `.babeirc` 文件（这是 babel 的配置文件）

```json
{
  "presets": ["es2015", "stage-2"], //设置转码规则
  "plugins": ["transform-runtime"] //设置转码插件
}
```

3. 这里我们需要转换成 es2015 ，安装我们需要的库：

```
npm install babel-core babel-preset-es2015 babel-plugin-transform-runtime babel-preset-stage-2 --save -dev
```

4. 在项目下的 `package.json` 文件做如下修改：

```json
"scripts":{"build":"babel src -w -d lib"},
```

即编译整个 src 目录并将其输出到 lib 目录。这里的 src 指的是需要转换的目录， lib 指的是输出的内容的存放目录， -w 其实是 -watch 的意思，就是监听文件，实时编译输出

5. 新建 src 目录和 lib 目录，不然会报错

6. 命令行输入 npm run build

## Promise

Promise是一门新的技术（ES6规范），用于进行异步编程。

**使用方法：**

```JavaScript
new Promise((resolve, reject) => {
    //代码片段
    resolve(res)
    reject(err)
})
    .then(res => {
        //代码片段
    }, err => {
        //代码片段
    })
```

`resolve` 调用该函数则代表执行成功，跳出 Promise ，以 res 为参数执行 `.then` 中的第一个函数

`reject` 调用该函数则代表执行失败，跳出 Promise ，以 err 为参数执行 `.then` 中的第二个函数

常规情况下 `resolve` 与 `reject` 最终只调用其中一个

**多层 Promise 调用**

```JavaScript
new Promise((resolve, reject) => {
    console.log('进入第一层')
    setTimeout(() => {
        resolve("进入第二层")
    }, 1000)
})
    .then(res => {
        console.log(res)
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve("进入第三层")
            }, 1000)
        })
    }, err => { })
    .then(res => {
        console.log(res)
    })
```

可以在最后使用 `.catch` 统一处理错误，而不用对每个 Promise 均编写处理错误的函数:

```JavaScript
new Promise((resolve, reject) => {
    console.log('进入第一层')
    setTimeout(() => {
        //reject("err")
        resolve("进入第二层")
    }, 1000)
})
    .then(res => {
        console.log(res)
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                reject("err")
                //resolve("进入第三层")
            }, 1000)
        })
    })
    .then(res => {
        console.log(res)
    })
    .catch(err => {
        console.log(err)
    })
```

**并行 Promise 调用**

```JavaScript
Promise.all([
    new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve('这是第一个请求')
        }, 1000)
    }),
    new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve('这是第二个请求')
        }, 1000)
    })
]).then(res => {
    console.log(res)
})
```

其中 res 的值为一个数组，其值分别代表不同 Promise 的 `resolve` 传递值
