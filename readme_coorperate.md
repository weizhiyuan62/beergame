# GitHub 小组协作说明

本文档用于统一 BeergameProject 小组成员在 GitHub 上的基本协作方式。请在提交代码、论文内容或实验结果前先阅读，并尽量遵守相同流程。

## 1. 基础配置
- git基础操作省略

```zsh
# clone the repo
git clone https://github.com/weizhiyuan62/BeergameProject-2026PKUSpring.git
cd BeergameProject-2026PKUSpring
git checkout main
```

## 2. 分支协作流程

- `main` 只保留已经确认可以合并的稳定内容。
- 每位成员固定使用自己的个人分支完成日常写作、代码和实验修改。
- 不要直接在 `main` 上提交修改。
- 个人分支合并到 `main` 时，必须在 GitHub 上发起 Pull Request。

```zsh
# ensure your main is the latest version
git checkout main
git pull origin main

# create your own branch
git checkout -b <你的名字>   # example: git checkout -b shouyuxin
git push origin <你的名字>   # example: git push origin shouyuxin
```

## 3. Commit Message 规范
```md
> format is below
<type>: <summary>
```
`type` 建议取值：
- `feat`：新功能（feature）。
- `fix/to`：修复bug，可以是QA发现的BUG，也可以是研发自己发现的BUG。
- `docs`：文档（documentation）。
- `style`：格式（不影响代码运行的变动）。
- `refactor`：重构（即不是新增功能，也不是修改bug的代码变动）。
- `revert`：回滚到上一个版本。
- `merge`：代码合并。
- `sync`：同步主线或分支的Bug。
- `paper`：对论文的修改增加之类的。

examples:

```zsh
git commit -m "paper: add beer game background section"
git commit -m "fix: remove invalid LaTeX reference"
git commit -m "docs: add GitHub collaboration guide"
```

## 4. Pull Request 合并流程

当认为自己的个人分支内容可以合并到 `main` 时，在 GitHub 上发起 Pull Request：
合并要求：
- 合并前建议至少由一名其他成员检查。(别人没空的话, 自己推, 自己通过也可以hhh)
- 如果涉及论文核心内容、实验结果或模型代码，建议检查。
- 如果出现冲突，先在本地解决冲突，再重新推送个人分支。
- 确认无误后，再将 Pull Request 合并到 `main`。


## 5. 谨慎注意的内容:
- 不要直接向 `main` 提交代码或论文内容。
- 不要使用 `git push --force` 覆盖远端分支，除非已经和组员确认。
- 不要提交 LaTeX 编译临时文件，例如 `.aux`、`.log`、`.out`、`.synctex.gz`。
- 不要提交大型模型文件、缓存文件或无关数据。
- 不要用 `final`, `new`, `latest`, `真的最终版` 这类文件名。