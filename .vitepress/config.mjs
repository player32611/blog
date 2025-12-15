import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/blog/",
  title: "我的个人笔记网页",
  description: "A VitePress Site",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '主页', link: '/' },
      { text: 'vitepress', link: '/pages/vitepress/quick-start' },
      { text: '前端', link: '/pages/front-end/html' },
      { text: 'GameMakerStudio 2', link: '/pages/gms2/gms2' },
      { text: '算法', link: '/pages/algorithm/c++-basic' },
    ],

    sidebar: [
      {
        text: 'vitepress使用指南',
        items: [
          { text: '快速开始', link: '/pages/vitepress/quick-start' },
          { text: '特殊语法', link: '/pages/vitepress/basic-grammer' }
        ]
      },
      {
        text: '前端',
        items: [
          { text: 'HTML', link: '/pages/front-end/html' },
          { text: 'CSS', link: '/pages/front-end/css' },
          { text: 'JavaScript', link: '/pages/front-end/javascript' },
          { text: 'DOM', link: '/pages/front-end/dom' },
          { text: 'npm与其他东西', link: '/pages/front-end/npm-and-others' },
          { text: '网络请求', link: '/pages/front-end/network-requests' },
          { text: 'webpack', link: '/pages/front-end/webpack' },
          { text: 'React', link: '/pages/front-end/react' },
          { text: 'Vue', link: '/pages/front-end/vue' },
          { text: '微信小程序', link: '/pages/front-end/wechat-miniprogram' },
        ]
      },
      {
        text: 'GameMakerStudio 2',
        items: [
          { text: '常用函数', link: '/pages/gms2/common-functions' }
        ]
      },
      {
        text: '算法',
        items: [
          { text: 'C++基础', link:'/pages/algorithm/c++-basic'},
          { text: 'STL' , link:'/pages/algorithm/stl'},
          { text: '数据结构', link:'/pages/algorithm/data-structure'},
          { text: '算法基础', link:'/pages/algorithm/algorithm-basic'}
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/player32611' },
      {
        icon: {
          svg: '<svg t="1750214089876" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4865" width="200" height="200"><path d="M184.32 0h655.36Q1024 0 1024 184.32v655.36Q1024 1024 839.68 1024H184.32Q0 1024 0 839.68V184.32Q0 0 184.32 0z" fill="#EC5D85" p-id="4866"></path><path d="M512 241.961h52.224l65.065-96.317c49.633-50.32 89.64 0.43 63.857 45.71l-34.315 51.508C916.48 247.89 916.48 285.87 916.48 567.9c0 325.95 0 336.466-404.48 336.466S107.52 893.85 107.52 567.9c0-277.698 0-318.802 253.143-324.956l-39.434-58.368c-31.263-54.907 37.335-90.409 64.686-42.373l60.416 99.81c18.186-0.052 41.185-0.052 65.669-0.052z" fill="#EF85A7" p-id="4867"></path><path d="M512 338.586c332.8 0 332.8 0 332.8 240.64s0 248.391-332.8 248.391-332.8-7.751-332.8-248.391 0-240.64 332.8-240.64z" fill="#EC5D85" p-id="4868"></path><path d="M281.6 558.08a30.72 30.72 0 0 1-27.474-16.978 30.72 30.72 0 0 1 13.732-41.216l122.88-61.44a30.72 30.72 0 0 1 41.216 13.742 30.72 30.72 0 0 1-13.742 41.216l-122.88 61.44a30.597 30.597 0 0 1-13.732 3.236z m471.04 0a30.607 30.607 0 0 1-12.851-2.836l-133.12-61.44a30.72 30.72 0 0 1-15.043-40.756 30.72 30.72 0 0 1 40.766-15.022l133.12 61.44a30.72 30.72 0 0 1-12.872 58.614z m-297.984 108.8a15.36 15.36 0 0 1-12.288-6.195 15.36 15.36 0 0 1 3.072-21.494l68.506-50.913 50.35 52.623a15.36 15.36 0 0 1-22.2 21.238l-31.591-33.024-46.715 34.724a15.288 15.288 0 0 1-9.134 3.041z" fill="#EF85A7" p-id="4869"></path><path d="M65.536 369.316c15.032 101.908 32.85 147.18 44.544 355.328 14.633 2.181 177.705 10.045 204.052-74.63a16.148 16.148 0 0 0 1.65-10.874c-30.608-80.333-169.217-60.416-169.217-60.416s-10.363-146.504-11.5-238.838z m296.714 13.721l34.816 303.176h34.642l-26.47-305.08zM309.524 536.29h45.487l16.097 158.618-31.826 1.853z m137.34 6.697h45.803v162.345h-33.874z m-150.262-85.012h21.391l5.274 58.992-18.914 2.263z m30.392 0h21.391l2.54 55.808-17.408 1.618z m143.626 1.915h19.456v62.28H470.62z m-30.382 0h22.2v62.28h-16.63z" fill="#FFFFFF" p-id="4870"></path><path d="M243.569 645.52a275.456 275.456 0 0 1-28.468 23.746 242.688 242.688 0 0 1-29.532 17.52 2.703 2.703 0 0 1-4.403-1.955 258.601 258.601 0 0 1-5.12-29.573c-1.413-12.186-1.956-25.682-2.16-36.363 0-0.307 0-2.508 3.01-1.904a245.924 245.924 0 0 1 34.222 9.574 257.024 257.024 0 0 1 32.358 15.176c0.523 0.256 2.52 1.485 0.093 3.778z" fill="#EB5480" p-id="4871"></path><path d="M513.29 369.316c15.033 101.908 32.85 147.18 44.544 355.328 14.633 2.181 177.705 10.045 204.053-74.63a16.148 16.148 0 0 0 1.648-10.874c-30.607-80.333-169.216-60.416-169.216-60.416s-10.363-146.504-11.5-238.838z m296.714 13.721l34.816 303.176h34.642l-26.47-305.08zM757.28 536.29h45.486l16.097 158.618-31.826 1.853z m137.339 6.697h45.803v162.345h-33.874z m-150.262-85.012h21.391l5.274 58.992-18.913 2.263z m30.392 0h21.392l2.539 55.808-17.408 1.618z m143.626 1.915h19.456v62.28h-19.456z m-30.382 0h22.2v62.28h-16.63z" fill="#FFFFFF" p-id="4872"></path><path d="M691.323 645.52a275.456 275.456 0 0 1-28.467 23.746 242.688 242.688 0 0 1-29.532 17.52 2.703 2.703 0 0 1-4.404-1.955 258.601 258.601 0 0 1-5.12-29.573c-1.413-12.186-1.956-25.682-2.16-36.363 0-0.307 0-2.508 3.01-1.904a245.924 245.924 0 0 1 34.222 9.574 257.024 257.024 0 0 1 32.359 15.176c0.522 0.256 2.519 1.485 0.092 3.778z" fill="#EB5480" p-id="4873"></path></svg>'
        },
        link: 'https://space.bilibili.com/495270127?spm_id_from=333.1007.0.0'
      },
      { icon: 'gamejolt', link: 'https://gamejolt.com/@player32611' },
    ],
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索文档",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            noResultsText: "无法找到相关结果",
            resetButtonTitle: "清除查询条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
            },
          },
        },
      },
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    }
  },
  vite: {
    resolve: {
      alias: [
        {
          find: /^.*\/VPSwitchAppearance\.vue$/,
          replacement: fileURLToPath(
            new URL('./theme/components/ThemeSwitch.vue', import.meta.url)
          )
        }
      ]
    }
  }
})
