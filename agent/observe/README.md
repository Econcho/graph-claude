# Runtime Observe

runtime observe 的默认目标是记录 node 的真实输入、真实输出，以及 node 之间的实际流转。

## trace.jsonl

`trace.jsonl` 是完整事件流。每条记录保留五个顶层字段：

```json
{
  "id": "...",
  "result_type": "node_output",
  "output_node": "llm",
  "input_node": "graph",
  "content": {
    "node": "llm",
    "output": {}
  }
}
```

默认情况下：

- `node_input.content.state` 保存该 node 收到的完整 state。
- `node_output.content.output` 保存该 node 返回的局部 state patch。
- `route_decision.content.state` 保存 route 做决策时看到的完整 state。

这些内容会先转换成 JSON 可序列化结构。字符串会做长度截断，避免 trace 文件被超长文本撑爆。

## trace.mmd

`trace.mmd` 是简化后的 Mermaid 流转图。

当前 Mermaid 只画实际执行过的 node 之间的流转：

```text
bootstrap -> context -> prompt -> llm -> finalize
```

如果发生工具调用，会看到闭环：

```text
llm -> tool_orchestrator -> tool_executor -> tool_result -> context
```

Mermaid 不画：

- `graph` 占位节点
- node 输入结果节点
- node 输出结果节点
- state 内容
- output 内容
- result id

这些细节只保留在 `trace.jsonl` 中。

## 默认观测点

默认开启：

- `node_input`：graph 将当前 state 输入到某个 node。
- `node_output`：某个 node 输出局部 state patch。
- `route_decision`：route 函数决定下一跳 node。
- `run_error`、`node_error`、`llm_error`、`tool_call_error`：异常复盘。

默认关闭：

- `run_start`、`run_end`：这两类事件容易和节点数据流重复。
- `llm_start`、`llm_end`：这是 LLM 内部细节，不是 node 间流转的主线。
- `tool_call_start`、`tool_call_end`、`tool_result_appended`：这是 tool 内部细节，默认通过 `tool_executor` 和 `tool_result` 的 node 输入输出观察。

## 输出文件

每次运行会写入：

```text
.runs/
  run_xxx/
    trace.jsonl
    trace.mmd
```

## 阅读单条 observation

可以用结果 id 或唯一 id 前缀打印某条 observation：

```powershell
uv run python -m agent.observe.trace_reader .runs/run_xxx/trace.jsonl b7d03baa
```

也可以在代码中调用：

```python
from agent.observe.trace_reader import print_observation

print_observation(".runs/run_xxx/trace.jsonl", "b7d03baa")
```

## 配置

默认配置在 `agent/observe/config.json`。

- `mode`：当前为 `dataflow`，表示默认关注节点间数据流。
- `hooks`：控制每个 observe hook 是否生效。
