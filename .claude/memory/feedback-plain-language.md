---
name: feedback-plain-language
description: "回答要口语、清楚、用词简单,别堆术语和复杂书面句——用户嫌我说话太绕"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 606a95c7-0a2c-44df-8c70-61a6b1f56252
---

**规则:回答用大白话,说清楚就行,别堆专业术语和复杂的书面长句。**

**Why:** 2026-05-25 用户明确反馈:"你用词太复杂了,我需要更口语更清晰的表达。" 之前几轮我用了一堆行话(MIQP / CVaR / SOCP / manageability / homotopy 等)和书面腔长句,读着累。

**How to apply:**
- 能用大白话就用大白话。
- 必须用的专业词(比如 MIQP),先用一句人话解释,别默认对方秒懂。
- 句子短一点,像聊天,不要写成论文/报告腔。
- 少用花哨的结构、比喻和排比;一层意思讲清就够。
- 跟全局 CLAUDE.md "简洁优先" 互补:那条管**长度**,这条管**说人话**。
- **`AskUserQuestion` 选项**也算"回答":label/description 不能塞模糊术语糊弄过。每个选项要讲清:**它到底是什么、跟其他选项的真区别、什么时候会出问题**。用户 2026-05-28 明确"全都解释清楚!你这样用这些模糊不清的名词很容易糊弄过去"。
