# .claude/ — 跨电脑的 Claude 上下文快照

这里是从开发机的 `~/.claude` 和 `~/code/CLAUDE.md` 拷过来的**只读快照**,目的:换电脑 / 换会话时,Claude 不用从零开始,直接有完整背景(个人偏好、workspace 细节、研究计划与进度)。

## 内容
- `CLAUDE.user.md` —— 用户全局偏好(原 `~/.claude/CLAUDE.md`):称呼「塔菲大人」、中文简洁、研究方向等。
- `CLAUDE.workspace.md` —— workspace 技术细节(原 `~/code/CLAUDE.md`)。
- `memory/` —— Claude 自动记忆的 `.md`(原 `~/.claude/projects/-home-boxuan/memory/`):
  - `sando-rgbd-plan.md` = **当前方向/决策/进度的权威记录,先读它**;
  - `MEMORY.md` = 索引;`feedback-*` = 工作风格;`sando-rgbd-*` = 测试清单 / 已知坑 / 规格。

## 在别的电脑怎么用
1. **最省事**:在 `sando_py` 目录里跑 Claude Code —— 它会自动读仓库根的 `CLAUDE.md`,那里已指向本目录,直接让 Claude 读 `.claude/memory/sando-rgbd-plan.md` 等即可上手。
2. **想恢复「自动记忆」机制**(让 Claude 像在原机一样自动加载 memory):把本目录的文件拷回新机的对应位置——
   ```bash
   cp .claude/CLAUDE.user.md      ~/.claude/CLAUDE.md
   # memory 路径里的 -home-boxuan 是「项目根路径」转义,新机若用户名/路径不同需相应改名:
   mkdir -p ~/.claude/projects/-home-boxuan/memory
   cp .claude/memory/*.md         ~/.claude/projects/-home-boxuan/memory/
   ```
   （`~/code/CLAUDE.md` 是否需要,取决于新机是否也用 `~/code` 这套工作区布局。）

## 维护
这些是**快照,不会自动同步**。开发机上 memory/CLAUDE.md 有更新后,重新 `cp` 进来再提交才会带到别的电脑。
> 注意:含个人邮箱 + 研究计划,**只 push 到私有仓库**。
