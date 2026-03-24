# Spring Boot

> Spring Boot 可以帮助我们非常快速的构建应用程序、简化开发、提高效率

::: danger 警告

该页面尚未完工!

:::

::: details 目录

[[toc]]

:::

## HTTP 协议

**HTTP 协议**（Hyper Text Transfer Protocol、超文本传输协议）规定了浏览器和服务其之间数据传输的规则。

::: tip HTTP 协议的特点

1. 基于 TCP 协议：面向连接，安全

2. 基于请求-响应模型的：一次请求对应一次响应

3. HTTP 协议是无状态的协议：对于事务处理没有记忆能力，每次请求-响应都是独立的。多次请求间不能共享数据，但速度快。

:::

### 请求协议

请求数据格式由三部分组成：**请求行**、**请求头**、**请求体**

**请求行**：请求数据第一行，包含请求方式、资源路径、协议

**请求头**：第二行开始，格式为 `key: value`

**请求体**：POST 请求，存放请求参数

::: tip GET 与 POST

请求方式 - GET：请求参数在请求行中，没有请求体，如：`/brand/findAll?name=POOP&status=1`。GET 请求大小在浏览器中是有限制的

请求方式 - POST：请求参数在请求体中，POST 请求大小是没有限制的

:::

::: tip 常见的请求头

|     请求头      |                                                            详细信息                                                             |
| :-------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
|      Host       |                                                          请求的主机名                                                           |
|   User-Agent    | 浏览器版本、例如 Chrome 浏览器的标识类似 Mozilla/5.0 ... Chrome/79，IE 浏览器的标志类似 Mozilla/5.0（Windows NT ...）like Gecko |
|     Accept      |                              表示浏览器能接收的资源类型，如 text/\*，image/\* 或者 \*/\* 表示所有                               |
| Accept-Language |                                     表示浏览器偏好的语言，服务器可以据此返回不同语言的网页                                      |
| Accept-Encoding |                                       表示浏览器可以支持的压缩类型，例如 gzip，deflate 等                                       |
|  Content-Type   |                                                         请求主体的类型                                                          |
| Content-Length  |                                                  请求主体的大小（单位：字节）                                                   |

:::

Web 服务器（Tomcat）对 HTTP 协议的请求数据进行解析，并进行了封装（HttpServletRequest），在调用 Controller 方法的时候传递给了该方法。这样，就使得程序员不必直接对协议进行操作，让 Web 开发更加便捷。

```java
package com.example.demo.controller;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RequestController {
    @RequestMapping("/request")
    public String request(HttpServletRequest  request) {
        // 获取请求方式
        String method = request.getMethod();
        System.out.println("method: " + method);
        // 获取请求 url 地址
        String url = request.getRequestURL().toString(); // http://localhost:8080/request
        System.out.println("url: " + url);
        String uri = request.getRequestURI(); // /request
        System.out.println("uri: " + uri);
        // 获取请求协议
        String protocol = request.getProtocol();
        System.out.println("protocol: " + protocol);
        // 获取请求参数 - name,age
        String name = request.getParameter("name");
        System.out.println("name: " + name);
        // 获取请求头 - Accept
        String accept = request.getHeader("Accept");
        System.out.println("accept: " + accept);

        return "OK";
    }
}
```

### 响应协议

响应数据格式由三部分组成：**响应行**、**请求头**、**请求体**

**响应行**：响应数据第一行，包含协议、状态码、描述

**响应头**：第二行开始，格式 `key: value`

**响应体**：最后一部分，存放响应数据

::: tip 常见的响应头

|      响应头      |                            详细信息                            |
| :--------------: | :------------------------------------------------------------: |
|   Content-Type   |     表示该响应内容的类型，例如 text/html，application/json     |
|  Content-Length  |                 表示该响应内容的长度（字节数）                 |
| Content-Encoding |                 表示该响应压缩算法，例如 gzip                  |
|  Cache-Control   | 指示客户端应如何缓存，例如 max-age=300 表示可以最多缓存 300 秒 |
|    Set-Cookie    |            告诉浏览器为当前页面所在的域设置 cookie             |

:::

Web 服务器对 HTTP 协议的响应数据进行了封装（HttpServletResponse），并在调用 Controller 方法的时候传递给了该方法。这样，就使得程序员不必直接对协议进行操作，让 Web 开发更加便捷。

```java
package com.example.demo.controller;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class ResponseController {
    @RequestMapping("/response")
    public void response(HttpServletResponse response) throws IOException {
        // 设置响应状态码
        response.setStatus(HttpServletResponse.SC_OK);
        // 设置响应头
        response.setHeader("name", "text/html;charset=utf-8");
        // 设置响应体
        response.getWriter().write("<h1>Hello World</h1>");
    }

    @RequestMapping("/response2")
    public ResponseEntity<String> response2(){
        return ResponseEntity
                .status(401)
                .header("name", "text/html;charset=utf-8")
                .body("<h1>Hello response</h1>");
    }
}
```

::: warning 注意

响应状态码和响应头如果没有特殊要求的话，通常不手动设定。服务器会根据请求处理的逻辑，自动设置响应状态码和响应头

:::

## 分层解耦

### 三层架构

为了使我们所定义的接口、类以及方法的复杂度更低，可读性更强，扩展性更好，便于项目的后期维护，基于此，在 Web 开发中有了**三层架构**：

**controller**：控制层，接收前端发送的请求，对请求进行处理，并响应数据（接收请求，响应数据）

**service**：业务逻辑层，处理具体的业务逻辑（业务逻辑处理）

**dao**：数据访问层（Data Access Object）（持久层），负责数据访问操作。包括数据的增、删、改、查（数据访问操作）

::: code-group

```java [UserDao.java]
// /main/java/com.xxx.xxx/dao/UserDao.java

public interface UserDao {
    // 加载用户数据
    public List<String> findAll();
}
```

```java [UserDaoImpl.java]
// /main/java/com.xxx.xxx/dao/impl/UserDaoImpl.java

public class UserDaoImpl implements UserDao {

    @Override
    public List<String> findAll(){
        InputStream in = this.getClass().getClassLoader().getResourceAsStream("user.txt");
        ArrayList<String> lines = IOUtils.reafLines(in, StandardCharsets.UTF_8,new ArrayList<>());
        return lines;
    }
}
```

```java [UserService.java]
// /main/java/com.xxx.xxx/service/UserService.java

public interface UserService {
    // 查询所有用户信息
    public List<User> findAll();
}
```

```java [UserServiceImpl.java]
// /main/java/com.xxx.xxx/service/impl/UserServiceImpl.java

public class UserServiceImpl implements UserService {

    private UserDao userDao = new UserDaoImpl();

    @Override
    public List<User> findAll(){
        // 调用 dao，获取数据
        List<User> lines = userDao.findAll();

        // 解析用户信息，封装为 User 对象 -> list 集合
        List<User> userList = lines.stream().map(line->{
            String[] parts = line.split(",");
            Integer id = Integer.parseInt(parts[0]);
            String username = parts[1];
            String password = parts[2];
            String name = parts[3];
            Integer age = Integer.parseInt(parts[4]);
            LocalDateTime updateTime = LocalDateTime.parse(parts[5], DateTimeFormatter.ofPattern("yyyy-MM-dd H"));
            return new User(id,username,password,name,age,updateTime);
        }).toList();

        return userList;
    }
}
```

```java [UserController.java]
// /main/java/com.xxx.xxx/controller/UserController.java

@RestController
public class UserController {

    private UserService userService = new UserServiceImpl();

    @RequestMapping("/list")
    public List<User> list() throws Exception{
        // 调用 service，获取数据
        List<User> userList = userService.findAll();

        // 返回数据(json)
        return userList;
    }
}
```

:::

### 分层解耦

**耦合**：衡量软件中各个层/各个模块的依赖关联程度

**内聚**：软件中各个功能模块内部的功能联系

::: tip 软件设计原则

高内聚低耦合

:::

**控制反转**（Inversion Of Control，IOC）：对象的创建控制权由程序自身转移到外部（容器），这种思想被称为控制反转。

**依赖注入**（Dependency Injection，DI）：容器为应用程序提供运行时，所依赖的资源，称之为依赖注入

**Bean对象**：IOC 容器中创建、管理的对象，称之为 Bean

::: tip 实现分层解耦的思路

- 将项目中的类交给 IOC 容器管理（IOC，控制反转）

- 应用程序运行时需要什么对象，直接依赖容器为其提供（DI，依赖注入）

:::

1. 通过 `@Component` 将 Dao 及 Service 层的实现类，交给 IOC 容器管理（是加在实现类上，而非接口上）

2. 通过 `@Autowired` 为 Controller 及 Service 注入运行时所依赖的对象

::: code-group

```java [UserServiceImpl.java] 1,4
@Component // 将当前类交给 IOC 容器管理
public class UserServiceImpl implements UserService {

    @Autowired // 应用程序运行时，会自动的查询该类型的 bean 对象，并赋值给该成员变量
    private UserDao userDao;

    @Override
    public List<User> findAll(){
        // 调用 dao，获取数据
        List<User> lines = userDao.findAll();

        // 解析用户信息，封装为 User 对象 -> list 集合
        List<User> userList = lines.stream().map(line->{
            String[] parts = line.split(",");
            Integer id = Integer.parseInt(parts[0]);
            String username = parts[1];
            String password = parts[2];
            String name = parts[3];
            Integer age = Integer.parseInt(parts[4]);
            LocalDateTime updateTime = LocalDateTime.parse(parts[5], DateTimeFormatter.ofPattern("yyyy-MM-dd H"));
            return new User(id,username,password,name,age,updateTime);
        }).toList();

        return userList;
    }
}
```

```java [UserDaoImpl.java] 1
@Component // 将当前类交给 IOC 容器管理
public class UserDaoImpl implements UserDao {

    @Override
    public List<String> findAll(){
        InputStream in = this.getClass().getClassLoader().getResourceAsStream("user.txt");
        ArrayList<String> lines = IOUtils.reafLines(in, StandardCharsets.UTF_8,new ArrayList<>());
        return lines;
    }
}
```

```java [UserController.java] 4,5
@RestController
public class UserController {

    @Autowired
    private UserService userService;

    @RequestMapping("/list")
    public List<User> list() throws Exception{
        // 调用 service，获取数据
        List<User> userList = userService.findAll();

        // 返回数据(json)
        return userList;
    }
}
```

:::
