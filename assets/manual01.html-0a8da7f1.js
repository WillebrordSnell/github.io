const e=JSON.parse('{"key":"v-629118d4","path":"/Tools/Git/01Manual/manual01.html","title":"1. 常用Git命令清单","lang":"zh-CN","frontmatter":{"order":1,"date":"2023-05-15T00:00:00.000Z","category":["Git"],"description":"1. 常用Git命令清单 一般来说，日常使用只要记住下图6个命令，就可以了。但是熟练使用，恐怕要记住60～100个命令。 几个专用名词的译名 Workspace：工作区 Index / Stage：暂存区 Repository：仓库区（或本地仓库） Remote：远程仓库 1 新建代码库 # 在当前目录新建一个Git代码库 $ git init # 新建一个目录，将其初始化为Git代码库 $ git init [project-name] # 下载一个项目和它的整个代码历史 $ git clone [url]","head":[["meta",{"property":"og:url","content":"https://github.com/WillebrordSnell/Tools/Git/01Manual/manual01.html"}],["meta",{"property":"og:site_name","content":" "}],["meta",{"property":"og:title","content":"1. 常用Git命令清单"}],["meta",{"property":"og:description","content":"1. 常用Git命令清单 一般来说，日常使用只要记住下图6个命令，就可以了。但是熟练使用，恐怕要记住60～100个命令。 几个专用名词的译名 Workspace：工作区 Index / Stage：暂存区 Repository：仓库区（或本地仓库） Remote：远程仓库 1 新建代码库 # 在当前目录新建一个Git代码库 $ git init # 新建一个目录，将其初始化为Git代码库 $ git init [project-name] # 下载一个项目和它的整个代码历史 $ git clone [url]"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:image","content":"https://github.com/WillebrordSnell/"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2023-10-19T07:29:59.000Z"}],["meta",{"name":"twitter:card","content":"summary_large_image"}],["meta",{"name":"twitter:image:alt","content":"1. 常用Git命令清单"}],["meta",{"property":"article:author","content":"Mr.R"}],["meta",{"property":"article:published_time","content":"2023-05-15T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2023-10-19T07:29:59.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"1. 常用Git命令清单\\",\\"image\\":[\\"https://github.com/WillebrordSnell/\\"],\\"datePublished\\":\\"2023-05-15T00:00:00.000Z\\",\\"dateModified\\":\\"2023-10-19T07:29:59.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"Mr.R\\",\\"url\\":\\"https://github.com/WillebrordSnell\\"}]}"]]},"headers":[{"level":2,"title":"1 新建代码库","slug":"_1-新建代码库","link":"#_1-新建代码库","children":[]},{"level":2,"title":"2 配置","slug":"_2-配置","link":"#_2-配置","children":[]},{"level":2,"title":"3 增加/删除文件","slug":"_3-增加-删除文件","link":"#_3-增加-删除文件","children":[]},{"level":2,"title":"4 代码提交","slug":"_4-代码提交","link":"#_4-代码提交","children":[]},{"level":2,"title":"5 分支","slug":"_5-分支","link":"#_5-分支","children":[]},{"level":2,"title":"6 标签","slug":"_6-标签","link":"#_6-标签","children":[]},{"level":2,"title":"7 查看信息","slug":"_7-查看信息","link":"#_7-查看信息","children":[]},{"level":2,"title":"8 远程同步","slug":"_8-远程同步","link":"#_8-远程同步","children":[]},{"level":2,"title":"9 撤销","slug":"_9-撤销","link":"#_9-撤销","children":[]},{"level":2,"title":"10 常用操作组合","slug":"_10-常用操作组合","link":"#_10-常用操作组合","children":[]}],"git":{"createdTime":1697700599000,"updatedTime":1697700599000,"contributors":[{"name":"WillebrordSnell","email":"799976781@qq.com","commits":1}]},"readingTime":{"minutes":5.94,"words":1782},"filePathRelative":"Tools/Git/01Manual/manual01.md","localizedDate":"2023年5月15日","excerpt":"<h1> 1. 常用Git命令清单</h1>\\n<p>一般来说，日常使用只要记住下图6个命令，就可以了。但是熟练使用，恐怕要记住60～100个命令。</p>\\n<figure><figcaption> </figcaption></figure>\\n<p>几个专用名词的译名</p>\\n<blockquote>\\n<ul>\\n<li>Workspace：工作区</li>\\n<li>Index / Stage：暂存区</li>\\n<li>Repository：仓库区（或本地仓库）</li>\\n<li>Remote：远程仓库</li>\\n</ul>\\n</blockquote>\\n<h2> 1 新建代码库</h2>\\n<div class=\\"language-bash line-numbers-mode\\" data-ext=\\"sh\\"><pre class=\\"language-bash\\"><code><span class=\\"token comment\\"># 在当前目录新建一个Git代码库</span>\\n$ <span class=\\"token function\\">git</span> init\\n\\n<span class=\\"token comment\\"># 新建一个目录，将其初始化为Git代码库</span>\\n$ <span class=\\"token function\\">git</span> init <span class=\\"token punctuation\\">[</span>project-name<span class=\\"token punctuation\\">]</span>\\n\\n<span class=\\"token comment\\"># 下载一个项目和它的整个代码历史</span>\\n$ <span class=\\"token function\\">git</span> clone <span class=\\"token punctuation\\">[</span>url<span class=\\"token punctuation\\">]</span>\\n</code></pre><div class=\\"line-numbers\\" aria-hidden=\\"true\\"><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div></div></div>","autoDesc":true}');export{e as data};
