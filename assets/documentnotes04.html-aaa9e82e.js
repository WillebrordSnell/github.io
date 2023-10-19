import{_ as e}from"./plugin-vue_export-helper-c27b6911.js";import{o as a,c as s,f as n}from"./app-0f503e6a.js";const i={},d=n(`<h1 id="_4-git分支管理-查看分支" tabindex="-1"><a class="header-anchor" href="#_4-git分支管理-查看分支" aria-hidden="true">#</a> 4. Git分支管理-查看分支</h1><h2 id="查看分支" tabindex="-1"><a class="header-anchor" href="#查看分支" aria-hidden="true">#</a> 查看分支</h2><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch
  iss53
* master  <span class="token comment"># 带星号*表示当前所在分支</span>
  testing
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p><code>git branch</code> 命令不只是可以创建与删除分支。 如果不加任何参数运行它，会得到当前所有分支的一个列表。</p><h2 id="查看每个分支的最后提交" tabindex="-1"><a class="header-anchor" href="#查看每个分支的最后提交" aria-hidden="true">#</a> 查看每个分支的最后提交</h2><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch <span class="token parameter variable">-v</span>
  iss53   93b412c fix javascript issue
* master  7a98805 Merge branch <span class="token string">&#39;iss53&#39;</span>
  testing 782fd34 <span class="token builtin class-name">test</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="查看已-未-合并的分支" tabindex="-1"><a class="header-anchor" href="#查看已-未-合并的分支" aria-hidden="true">#</a> 查看已(未)合并的分支</h2><p><code>--merged</code> 与 <code>--no-merged</code> 这两个选项可以查看哪些分支已经合并或未合并到 <strong>当前</strong> 分支。</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch <span class="token parameter variable">--merged</span> <span class="token comment"># 查看已合并分支列表</span>
  iss53
* master
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>上面列表中分支名字前没有 <code>*</code> 号的分支通常可以使用 <code>git branch -d</code> 删除掉；</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch --no-merged <span class="token comment"># 查看未合并的分支列表</span>
  testing
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><p>上面显示未合并的分支，尝试使用 <code>git branch -d</code> 命令删除它时会失败：</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch <span class="token parameter variable">-d</span> testing
error: The branch <span class="token string">&#39;testing&#39;</span> is not fully merged.
If you are sure you want to delete it, run <span class="token string">&#39;git branch -D testing&#39;</span><span class="token builtin class-name">.</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>强制删除未合并的分支:</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token function">git</span> branch <span class="token parameter variable">-D</span> testing
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h4 id="查看指定分支的已-未-合并的分支" tabindex="-1"><a class="header-anchor" href="#查看指定分支的已-未-合并的分支" aria-hidden="true">#</a> 查看指定分支的已(未)合并的分支</h4><p>上面描述的选项 <code>--merged</code> 和 <code>--no-merged</code> 会在没有给定提交或分支名作为参数时， 分别列出已合并或未合并到 <strong>当前</strong> 分支的分支。</p><p>你总是可以提供一个附加的参数来查看其它分支的合并状态而不必检出它们。 例如，尚未合并到 <code>testing</code> 分支的有哪些？</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>$ <span class="token function">git</span> branch --no-merged testing
  topicA
  featureB
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,19),r=[d];function c(t,l){return a(),s("div",null,r)}const h=e(i,[["render",c],["__file","documentnotes04.html.vue"]]);export{h as default};
