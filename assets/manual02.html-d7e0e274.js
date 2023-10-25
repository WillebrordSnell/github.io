const e=JSON.parse(`{"key":"v-6445f173","path":"/Tools/Git/01Manual/manual02.html","title":"2. Git变基合并","lang":"zh-CN","frontmatter":{"order":2,"date":"2023-05-15T00:00:00.000Z","category":["Git"],"description":"2. Git变基合并 1 rebase分支合并 1.1 说明 以下 v2 是某个需求的开发分支， dev是总的开发分支，v2 是基于dev分支签出的。 当完成v2的开发后，需要把代码合并到dev，我们可以使用rebase进行合并： # 首先将 v2 push到远程仓库 git add . git commit -m 'xxx' git push origin v2 # 切换到 dev 拉取最新代码 git checkout dev git pull origin dev # 切换到 v2 git checkout v2 git rebase dev # 将 v2 的所有[commit] 变基到(应用到) dev # 切换到 dev git checkout dev git merge v2 # 将 dev分支 快进合并 （此时 (HEAD -&gt; dev, v2) [commit] 两个分支指向同一个提交） # 查看 原v2的[commit]记录 是否在dev分支的最前面（变基成功会把v2的提交记录应用到dev分支的最前面） git log # 如果到这一步发现有问题，尝试使用 git --abort中止变基，如果还是有问题的可以在dev分支上使用《后悔药》操作， 再到v2分支上使用《后悔药》操作，即可使两个分支都回退到 rebase变基 之前的状态 # 试运行项目是否有问题 yarn start git status # 查看状态是否有问题 git push origin dev # 推送到远程仓库的 dev","head":[["meta",{"property":"og:url","content":"https://github.com/WillebrordSnell/Tools/Git/01Manual/manual02.html"}],["meta",{"property":"og:site_name","content":" "}],["meta",{"property":"og:title","content":"2. Git变基合并"}],["meta",{"property":"og:description","content":"2. Git变基合并 1 rebase分支合并 1.1 说明 以下 v2 是某个需求的开发分支， dev是总的开发分支，v2 是基于dev分支签出的。 当完成v2的开发后，需要把代码合并到dev，我们可以使用rebase进行合并： # 首先将 v2 push到远程仓库 git add . git commit -m 'xxx' git push origin v2 # 切换到 dev 拉取最新代码 git checkout dev git pull origin dev # 切换到 v2 git checkout v2 git rebase dev # 将 v2 的所有[commit] 变基到(应用到) dev # 切换到 dev git checkout dev git merge v2 # 将 dev分支 快进合并 （此时 (HEAD -&gt; dev, v2) [commit] 两个分支指向同一个提交） # 查看 原v2的[commit]记录 是否在dev分支的最前面（变基成功会把v2的提交记录应用到dev分支的最前面） git log # 如果到这一步发现有问题，尝试使用 git --abort中止变基，如果还是有问题的可以在dev分支上使用《后悔药》操作， 再到v2分支上使用《后悔药》操作，即可使两个分支都回退到 rebase变基 之前的状态 # 试运行项目是否有问题 yarn start git status # 查看状态是否有问题 git push origin dev # 推送到远程仓库的 dev"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2023-10-19T07:29:59.000Z"}],["meta",{"property":"article:author","content":"Mr.R"}],["meta",{"property":"article:published_time","content":"2023-05-15T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2023-10-19T07:29:59.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"2. Git变基合并\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2023-05-15T00:00:00.000Z\\",\\"dateModified\\":\\"2023-10-19T07:29:59.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"Mr.R\\",\\"url\\":\\"https://github.com/WillebrordSnell\\"}]}"]]},"headers":[{"level":2,"title":"1 rebase分支合并","slug":"_1-rebase分支合并","link":"#_1-rebase分支合并","children":[{"level":3,"title":"1.1 说明","slug":"_1-1-说明","link":"#_1-1-说明","children":[]},{"level":3,"title":"1.2 变基要遵守的准则","slug":"_1-2-变基要遵守的准则","link":"#_1-2-变基要遵守的准则","children":[]},{"level":3,"title":"1.3 变基的实质","slug":"_1-3-变基的实质","link":"#_1-3-变基的实质","children":[]}]},{"level":2,"title":"2 后悔药","slug":"_2-后悔药","link":"#_2-后悔药","children":[]},{"level":2,"title":"3 开发期间的rebase操作","slug":"_3-开发期间的rebase操作","link":"#_3-开发期间的rebase操作","children":[{"level":3,"title":"3.1 背景","slug":"_3-1-背景","link":"#_3-1-背景","children":[]},{"level":3,"title":"3.2 操作步骤","slug":"_3-2-操作步骤","link":"#_3-2-操作步骤","children":[]}]},{"level":2,"title":"3 git cherry-pick","slug":"_3-git-cherry-pick","link":"#_3-git-cherry-pick","children":[{"level":3,"title":"3.1 基本应用","slug":"_3-1-基本应用","link":"#_3-1-基本应用","children":[]},{"level":3,"title":"3.2 转移多个提交","slug":"_3-2-转移多个提交","link":"#_3-2-转移多个提交","children":[]}]}],"git":{"createdTime":1697700599000,"updatedTime":1697700599000,"contributors":[{"name":"WillebrordSnell","email":"799976781@qq.com","commits":1}]},"readingTime":{"minutes":4.29,"words":1288},"filePathRelative":"Tools/Git/01Manual/manual02.md","localizedDate":"2023年5月15日","excerpt":"<h1> 2. Git变基合并</h1>\\n<h2> 1 rebase分支合并</h2>\\n<h3> 1.1 说明</h3>\\n<p>以下 <code>v2</code> 是某个需求的开发分支， <code>dev</code>是总的开发分支，<code>v2</code> 是基于<code>dev</code>分支签出的。</p>\\n<p>当完成<code>v2</code>的开发后，需要把代码合并到<code>dev</code>，我们可以使用<code>rebase</code>进行合并：</p>\\n<div class=\\"language-bash line-numbers-mode\\" data-ext=\\"sh\\"><pre class=\\"language-bash\\"><code><span class=\\"token comment\\"># 首先将 v2 push到远程仓库</span>\\n<span class=\\"token function\\">git</span> <span class=\\"token function\\">add</span> <span class=\\"token builtin class-name\\">.</span>\\n<span class=\\"token function\\">git</span> commit <span class=\\"token parameter variable\\">-m</span> <span class=\\"token string\\">'xxx'</span>\\n<span class=\\"token function\\">git</span> push origin v2\\n\\n<span class=\\"token comment\\"># 切换到 dev 拉取最新代码</span>\\n<span class=\\"token function\\">git</span> checkout dev\\n<span class=\\"token function\\">git</span> pull origin dev\\n\\n<span class=\\"token comment\\"># 切换到 v2</span>\\n<span class=\\"token function\\">git</span> checkout v2\\n<span class=\\"token function\\">git</span> rebase dev <span class=\\"token comment\\"># 将 v2 的所有[commit] 变基到(应用到) dev</span>\\n\\n<span class=\\"token comment\\"># 切换到 dev</span>\\n<span class=\\"token function\\">git</span> checkout dev\\n<span class=\\"token function\\">git</span> merge v2  <span class=\\"token comment\\"># 将 dev分支 快进合并 （此时 (HEAD -&gt; dev, v2) [commit] 两个分支指向同一个提交）</span>\\n\\n<span class=\\"token comment\\"># 查看 原v2的[commit]记录 是否在dev分支的最前面（变基成功会把v2的提交记录应用到dev分支的最前面）</span>\\n<span class=\\"token function\\">git</span> log\\n\\n<span class=\\"token comment\\"># 如果到这一步发现有问题，尝试使用 git --abort中止变基，如果还是有问题的可以在dev分支上使用《后悔药》操作， 再到v2分支上使用《后悔药》操作，即可使两个分支都回退到 rebase变基 之前的状态</span>\\n\\n<span class=\\"token comment\\"># 试运行项目是否有问题</span>\\n<span class=\\"token function\\">yarn</span> start\\n\\n<span class=\\"token function\\">git</span> status <span class=\\"token comment\\"># 查看状态是否有问题</span>\\n<span class=\\"token function\\">git</span> push origin dev <span class=\\"token comment\\"># 推送到远程仓库的 dev</span>\\n\\n</code></pre><div class=\\"line-numbers\\" aria-hidden=\\"true\\"><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div><div class=\\"line-number\\"></div></div></div>","autoDesc":true}`);export{e as data};
