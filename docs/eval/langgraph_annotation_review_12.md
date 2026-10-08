# LangGraph Claim–Evidence 第一批人工审核表

> 状态：候选标注，尚未经人工确认；不能作为正式 benchmark 的 gold 数据。

## 来源和规则

- 依据上传的 `langgraph_passages.json` 与 `retrieval_bilingual_v1_draft.json`（还原粘贴过程的额外转义后）生成。
- 3 篇网页的 Passage 切分与正文快照共 245/245 一致。
- 证据保留 Passage 的**完整精确文本**，不改写 Gold Quote。
- `cited_urls` 是 benchmark 模拟的允许检索来源池，不等同于真实模型回答已经引用了这些 URL。
- 对多 Gold 样本，请逐条判断是否**独立**足以支持整条 Claim；需要结合上下文才成立的候选应删除或修订。

## 人工审核清单

| 审核 | ID   | 语言 | 主题        | Claim                                                        | Gold Passage ID      | 备注                                                         |
| ---- | ---- | ---- | ----------- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| ☐    | Q001 | zh   | Overview    | LangGraph 专注于底层 Agent 编排。                            | D01_P0004            | 定义：底层编排                                               |
| ☐    | Q002 | en   | Overview    | LangGraph can combine deterministic hand-coded steps and LLM-driven agentic steps in one graph. | D01_P0003, D01_P0009 | 两个段落均直接陈述同图混合能力                               |
| ☐    | Q003 | zh   | Overview    | 使用 LangGraph 并不强制要求同时使用 LangChain。              | D01_P0006, D01_P0039 | 两处均明确声明可独立使用                                     |
| ☐    | Q004 | en   | Overview    | LangGraph provides durable execution, streaming, and human-in-the-loop as orchestration capabilities. | D01_P0008, D01_P0013 | 两个段落分别列出核心能力                                     |
| ☐    | Q005 | zh   | Persistence | Checkpointer 将线程的图状态持久化为 checkpoint。             | D02_P0003            | 单个 checkpointer 的线程范围                                 |
| ☐    | Q006 | en   | Persistence | Stores persist application-defined data outside the graph state. | D02_P0005            | 将 Store 与 Checkpointer 区分                                |
| ☐    | Q007 | zh   | Persistence | Store 可用于跨线程持久化长期数据。                           | D02_P0006, D02_P0022 | P0006 的 them 需要结合相邻 P0005 的指代审核                  |
| ☐    | Q008 | en   | Persistence | The documented fix for overly long thread IDs is to keep thread_id values under 255 characters. | D02_P0014            | 源文是 troubleshooting guidance，不应泛化为所有数据库的固有限制 |
| ☐    | Q009 | zh   | Interrupts  | 中断可以暂停图执行，并等待外部输入后再继续。                 | D03_P0001            | 中断概述                                                     |
| ☐    | Q010 | en   | Interrupts  | When an interrupt is triggered, LangGraph saves graph state and waits for execution to resume. | D03_P0003, D03_P0012 | P0012 是调用 interrupt 的等价表达                            |
| ☐    | Q011 | zh   | Interrupts  | interrupt() 函数可以在图节点中的任意位置调用。               | D03_P0004            | 节点内任意位置                                               |
| ☐    | Q012 | en   | Interrupts  | The resume value is passed back to the interrupt call, allowing the node to continue with external input. | D03_P0018            | 恢复值传播                                                   |

## Gold 原文核对

### Q001 — LangGraph 专注于底层 Agent 编排。

- 允许来源：overview

- 当前状态：`candidate_pending_human_review`

- **D01_P0004** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > LangGraph is very low-level, and focused entirely on agent orchestration .

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q002 — LangGraph can combine deterministic hand-coded steps and LLM-driven agentic steps in one graph.

- 允许来源：overview, persistence

- 当前状态：`candidate_pending_human_review`

- **D01_P0003** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > LangGraph gives you fine-grained control to mix deterministic, hand-coded steps with LLM-driven agentic steps in the same graph, so you can build bespoke agents that behave exactly the way your application requires.

- **D01_P0009** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > One of LangGraph’s core strengths is the ability to mix deterministic steps with LLM-driven agentic steps in a single graph.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q003 — 使用 LangGraph 并不强制要求同时使用 LangChain。

- 允许来源：overview

- 当前状态：`candidate_pending_human_review`

- **D01_P0006** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > We will commonly use LangChain components throughout the documentation to integrate models and tools, but you don’t need to use LangChain to use LangGraph.

- **D01_P0039** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q004 — LangGraph provides durable execution, streaming, and human-in-the-loop as orchestration capabilities.

- 允许来源：overview, interrupts

- 当前状态：`candidate_pending_human_review`

- **D01_P0008** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > LangGraph is focused on the underlying capabilities important for agent orchestration: durable execution, streaming, human-in-the-loop, and more.

- **D01_P0013** — `https://docs.langchain.com/oss/python/langgraph/overview`

  > LangGraph is the orchestration runtime: durable execution, streaming, human-in-the-loop, and persistence.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q005 — Checkpointer 将线程的图状态持久化为 checkpoint。

- 允许来源：persistence

- 当前状态：`candidate_pending_human_review`

- **D02_P0003** — `https://docs.langchain.com/oss/python/langgraph/persistence`

  > LangGraph provides two complementary persistence systems: Checkpointers persist a thread’s graph state as checkpoints.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q006 — Stores persist application-defined data outside the graph state.

- 允许来源：persistence, overview

- 当前状态：`candidate_pending_human_review`

- **D02_P0005** — `https://docs.langchain.com/oss/python/langgraph/persistence`

  > Stores persist application-defined data outside the graph state.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q007 — Store 可用于跨线程持久化长期数据。

- 允许来源：persistence

- 当前状态：`candidate_pending_human_review`

- **D02_P0006** — `https://docs.langchain.com/oss/python/langgraph/persistence`

  > Use them for long-term, cross-thread memory, including user preferences, facts, and shared knowledge.

- **D02_P0022** — `https://docs.langchain.com/oss/python/langgraph/persistence`

  > Use stores to persist durable data across threads.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q008 — The documented fix for overly long thread IDs is to keep thread_id values under 255 characters.

- 允许来源：persistence, interrupts

- 当前状态：`candidate_pending_human_review`

- **D02_P0014** — `https://docs.langchain.com/oss/python/langgraph/persistence`

  > Fix: Keep thread_id values under 255 characters.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q009 — 中断可以暂停图执行，并等待外部输入后再继续。

- 允许来源：interrupts

- 当前状态：`candidate_pending_human_review`

- **D03_P0001** — `https://docs.langchain.com/oss/python/langgraph/interrupts`

  > Interrupts allow you to pause graph execution at specific points and wait for external input before continuing.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q010 — When an interrupt is triggered, LangGraph saves graph state and waits for execution to resume.

- 允许来源：interrupts, persistence

- 当前状态：`candidate_pending_human_review`

- **D03_P0003** — `https://docs.langchain.com/oss/python/langgraph/interrupts`

  > When an interrupt is triggered, LangGraph saves the graph state using its persistence layer and waits indefinitely until you resume execution.

- **D03_P0012** — `https://docs.langchain.com/oss/python/langgraph/interrupts`

  > When you call interrupt within a node, LangGraph saves the current graph state and waits for you to resume execution with input.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q011 — interrupt() 函数可以在图节点中的任意位置调用。

- 允许来源：interrupts

- 当前状态：`candidate_pending_human_review`

- **D03_P0004** — `https://docs.langchain.com/oss/python/langgraph/interrupts`

  > Interrupts work by calling the interrupt() function at any point in your graph nodes.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

### Q012 — The resume value is passed back to the interrupt call, allowing the node to continue with external input.

- 允许来源：interrupts, overview

- 当前状态：`candidate_pending_human_review`

- **D03_P0018** — `https://docs.langchain.com/oss/python/langgraph/interrupts`

  > The resume value is passed back to the interrupt call, allowing the node to continue execution with the external input.

- 人工结论：[ ] 通过  [ ] 需修改  [ ] 删除；备注：________

## 优先核对的风险

- **Q007 / D02_P0006：** Passage 以 `Use them` 开头；独立阅读有指代歧义。若要求每个 Gold 单段自足，保留 D02_P0022 即可。
- **Q008 / D02_P0014：** 数值修复建议来自 PostgresSaver 故障排除上下文，不应解释成所有存储后端的通用规定。
- **Q003、Q004、Q010：** 多个 Gold 是否各自足以支持 Claim，请独立判断。
- 正文解析可能包含 HTML 导航或代码切分噪音，不符合标准的 passage 应在标注时剔除，而不是人工改写 `gold_quote`。

## 审核完成后

把确认的 `gold_evidence` 项目的 `review_status` 改为 `approved`，拒绝项删去；若一条 Claim 已无 approved Gold，则整条暂不纳入正式评测。该审核表不等同于语义自动核验结果。