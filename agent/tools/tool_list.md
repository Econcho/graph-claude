# Tool List

本文档列出当前 agent 已支持的工具、用途、输入格式、安全属性和典型用法。

所有工具都实现统一 `Tool Protocol`，经由：

```text
ToolRegistry -> PromptRuntime tool specs -> LLM tool_call -> ToolExecutor -> ToolResult -> ToolMessage
```

工具安全属性说明：

| 字段 | 含义 |
| --- | --- |
| `read_only` | 是否只读。只读工具在 `plan` mode 下允许执行。 |
| `destructive` | 是否可能修改 workspace、执行命令或产生副作用。默认权限模式下通常需要 ask。 |
| `concurrency_safe` | 是否适合并发执行。当前 orchestrator 仍是 FIFO，但该字段为后续并发调度预留。 |

默认权限规则在 `agent/settings.json` 中配置，执行优先级是：

```text
deny > ask > allow > permission_mode fallback
```

---

## Builtin Coding Tools

这些工具来自 `agent/tools/builtin.py`，是主 agent 默认可见的基础 coding tools。

## File Tools

### `read_file`

读取 workspace 内 UTF-8 文本文件。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "path": "src/main.py"
}
```

输出：

```text
文件内容
```

典型用途：

- 读取源码文件
- 读取 README / 配置文件
- 验证 `write_file` 或 `edit_file` 的结果

注意：

- 只能读取 workspace 内路径。
- 默认 settings deny `.env` / `.env.*`。

### `read_many_files`

批量读取多个 workspace 内 UTF-8 文本文件。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "paths": ["src/a.py", "src/b.py"],
  "max_files": 20,
  "max_chars_per_file": 8000
}
```

输出：

```json
{
  "files": [
    {
      "path": "src/a.py",
      "ok": true,
      "content": "...",
      "truncated": false
    }
  ]
}
```

典型用途：

- 一次读取相关模块
- 对比多个文件实现
- 为修改前收集上下文

### `write_file`

写入 UTF-8 文本文件，可新建文件或覆盖已有文件。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "path": "SUMMARY.md",
  "content": "# Summary\n...",
  "overwrite": false
}
```

行为：

- 文件不存在时创建。
- 文件已存在且 `overwrite=false` 时失败。
- 父目录不存在时自动创建。

典型用途：

- 创建新文档
- 写生成文件
- 小型 demo 中创建输出文件

注意：

- 修改已有代码文件更推荐使用 `edit_file`。
- 默认 settings allow `write_file`，但 deny `.env*`、`secrets/**`、`.git/**` 等敏感路径。

### `edit_file`

对已有 UTF-8 文件应用精确文本替换。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "path": "src/app.py",
  "edits": [
    {
      "old": "return old_value",
      "new": "return new_value"
    }
  ]
}
```

行为：

- 文件必须存在。
- 每个 `old` 必须精确匹配一次。
- 匹配 0 次返回 `old_text_not_found`。
- 匹配多次返回 `old_text_not_unique`。
- 成功后返回 unified diff。

典型用途：

- 修改已有代码
- 小范围重构
- 修复测试失败

### `mkdir`

创建 workspace 内目录。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "path": "docs/architecture"
}
```

典型用途：

- 创建新模块目录
- 创建文档目录

### `move_file`

移动或重命名 workspace 内文件。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "source": "old/path.py",
  "destination": "new/path.py",
  "overwrite": false
}
```

行为：

- source 必须是文件。
- destination 已存在且 `overwrite=false` 时失败。

### `delete_file`

删除 workspace 内文件。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "path": "tmp/output.txt"
}
```

行为：

- 只删除文件，不删除目录。
- 默认应走 ask 或 deny。

---

## Search Tools

### `list_dir`

列出 workspace 内目录项。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "path": ".",
  "recursive": false,
  "max_entries": 200
}
```

输出项包含：

```json
{
  "path": "src/app.py",
  "type": "file",
  "size": 1234
}
```

默认跳过：

```text
.git
.venv
__pycache__
node_modules
.pytest_cache
.runs
dist
build
```

### `glob`

按相对 glob pattern 查找 workspace 文件。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "pattern": "src/**/*.py",
  "max_results": 200
}
```

注意：

- pattern 必须是 workspace 相对路径。
- 不允许绝对路径或 `..`。

### `grep`

在 workspace 文本文件中搜索正则。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "pattern": "class Agent",
  "path": ".",
  "case_sensitive": false,
  "max_matches": 100
}
```

输出：

```text
agent/graph/state.py:8: class AgentState(...)
```

行为：

- 跳过二进制文件。
- 跳过大文件。
- 默认跳过缓存目录。

---

## Command Tools

### `run_command`

在 workspace cwd 中执行 shell 命令。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "command": "uv run python -m compileall agent tests",
  "shell": "powershell",
  "timeout_seconds": 30
}
```

输出：

```json
{
  "returncode": 0,
  "stdout": "...",
  "stderr": "...",
  "shell": "powershell"
}
```

注意：

- Windows 默认 `powershell`。
- POSIX 默认 `bash`。
- stdout/stderr 会截断。
- 默认权限模式下需要 ask。
- `plan` mode 下 deny。

### `run_tests`

运行项目测试命令，并解析测试摘要。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "command": "uv run pytest",
  "shell": "powershell",
  "timeout_seconds": 120
}
```

输出：

```json
{
  "returncode": 0,
  "summary": "99 passed, 1 skipped in 19.75s",
  "stdout": "...",
  "stderr": "..."
}
```

典型用途：

- 修改代码后验证回归
- 运行目标测试文件
- 面试 observe 中展示“修改 -> 测试闭环”

---

## Git Tools

Git 工具都是只读工具，不执行 commit、reset、checkout、push 等修改操作。

### `git_status`

查看 workspace Git 状态。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{}
```

输出：

```text
 M README.md
?? tests/test_new.py
```

非 Git repo 返回：

```text
error=not_git_repository
```

### `git_diff`

查看 workspace diff。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "path": "agent/tools/builtin.py",
  "staged": false,
  "max_chars": 12000
}
```

典型用途：

- 总结本次修改
- 避免覆盖用户已有改动
- 面试展示变更复盘

### `git_log`

查看最近 commit。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "max_count": 10
}
```

### `git_show`

查看指定 Git revision/object。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "rev": "HEAD",
  "max_chars": 12000
}
```

---

## Diagnostic Tools

### `parse_test_output`

把测试输出解析为结构化 diagnostics。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "output": "FAILED tests/test_a.py::test_x - AssertionError: bad",
  "kind": "pytest"
}
```

输出：

```json
{
  "diagnostics": [
    {
      "file": "tests/test_a.py",
      "line": null,
      "test": "test_x",
      "message": "AssertionError: bad"
    }
  ]
}
```

### `collect_diagnostics`

执行诊断命令并解析输出。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "command": "uv run pytest tests/test_a.py",
  "shell": "powershell",
  "timeout_seconds": 120
}
```

典型用途：

- 运行测试并提取失败位置
- 把 raw output 转成下一轮可修复的结构化错误

---

## Symbol Tools

当前符号工具是启发式文本实现，不依赖 LSP/tree-sitter。返回结果会标记：

```text
confidence=heuristic
```

### `outline_file`

提取文件结构摘要。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "path": "src/math_utils.py"
}
```

支持：

- Python：`def`、`class`
- JS/TS：`function`、`class`、`const name =`、`export`
- Markdown：heading

输出：

```text
1: class Calculator
4: function add
```

### `find_definition`

查找 symbol 定义候选。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "symbol": "add",
  "path": "src"
}
```

输出：

```json
{
  "matches": [
    {
      "path": "src/math_utils.py",
      "line": 4,
      "preview": "def add(a, b):",
      "confidence": "heuristic"
    }
  ]
}
```

### `find_references`

查找 symbol 引用候选。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "symbol": "add",
  "path": "."
}
```

---

## Skill Tools

这些工具来自 `agent/skills/tools.py`，通过 `build_graph()` 加入主 ToolRegistry。

### `use_skill`

加载并执行指定 Skill。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "name": "skill-name",
  "args": "optional free-form arguments"
}
```

行为：

- 根据 Skill name 查找 `SKILL.md`。
- 展开 skill prompt。
- `context=inline` 时直接返回 prompt。
- `context=fork` 时启动隔离 skill subagent。

注意：

- PromptRuntime 只有在存在可用 skill summaries 时才暴露 `use_skill`。

### `shell_command`

Skill prompt shell 专用命令执行工具。

安全属性：

```text
read_only=false
destructive=true
concurrency_safe=false
```

输入：

```json
{
  "command": "echo hello",
  "shell": "powershell"
}
```

注意：

- 该工具默认不作为普通 main-agent shell 暴露。
- PromptRuntime 只有在 `active_skill_name` 非空时才暴露 `shell_command`。
- 普通命令执行应使用 `run_command`。

---

## Multi-Agent Tools

这些工具来自 `agent/multi_agent/tools.py`，通过 `collaboration_tools(manager)` 加入 ToolRegistry。

### `agent`

启动隔离 subagent 或 teammate。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "description": "Investigate failing tests",
  "prompt": "Read the failing test and propose a fix.",
  "run_in_background": false,
  "name": "worker-a",
  "team_name": "team-demo",
  "mode": "optional"
}
```

行为：

- 无 `team_name/name` 时创建普通 subagent。
- 有 `team_name + name` 时创建 teammate。
- `run_in_background=true` 时返回 `task_id`，后续通过 `agent_poll` 查询。

### `agent_poll`

查询 background agent task 状态。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{
  "task_id": "task_xxx"
}
```

不传 `task_id` 时列出所有当前进程内 task。

### `send_message`

向 teammate inbox 发送消息。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "to": "worker-a",
  "message": "Please inspect src/app.py"
}
```

广播：

```json
{
  "to": "*",
  "message": "Standup update"
}
```

### `team_create`

创建 workspace-local team。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "team_name": "demo-team",
  "leader": "leader"
}
```

写入：

```text
<workspace>/.agent/teams/<team>/team.json
<workspace>/.agent/teams/<team>/tasks.json
```

### `task_create`

创建共享 team task。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "description": "Fix failing parser test"
}
```

### `task_list`

列出共享 team tasks。

安全属性：

```text
read_only=true
destructive=false
concurrency_safe=true
```

输入：

```json
{}
```

### `task_update`

更新共享 team task 的状态或 assignee。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "task_id": "team_task_xxx",
  "status": "in_progress",
  "assignee": "worker-a"
}
```

### `task_stop`

把共享 team task 标记为 stopped。

安全属性：

```text
read_only=false
destructive=false
concurrency_safe=false
```

输入：

```json
{
  "task_id": "team_task_xxx"
}
```

---

## MCP Tools

MCP tools 不是固定写死的 Python 类，而是由 `McpManager.discover_tools()` 从 MCP server `tools/list` 动态发现，再通过 `McpToolAdapter` 转成内部 Tool Protocol。

命名规则：

```text
mcp__{server_name}__{tool_name}
```

示例：

```text
mcp__time__get_current_time
mcp__time__convert_time
mcp__fetch__fetch
mcp__github__get_file_contents
```

安全属性：

```text
read_only = MCP annotations.readOnlyHint
destructive = not read_only
concurrency_safe = false
```

当前 `agent/settings.json` 默认启用：

```text
time
fetch
```

GitHub MCP 已预留配置，但默认 disabled。

典型调用：

```json
{
  "name": "mcp__fetch__fetch",
  "args": {
    "url": "https://example.com"
  }
}
```

注意：

- MCP tool 仍然走 `ToolExecutor` 权限判断。
- 缺少 readOnlyHint 的 MCP tool 默认视为 destructive。

---

## Internal Memory Tool

### `edit_session_memory`

该工具由 session memory fork agent 内部使用，不属于主 agent 默认 builtin tools。

用途：

- 写入完整更新后的 session memory markdown。

输入：

```json
{
  "updated_session_memory_markdown": "# Session Memory\n..."
}
```

注意：

- 由 `SessionMemoryForkAgent` 使用。
- 不建议暴露给普通主 agent。

---

## Tool Registration Checklist

新增普通工具时需要：

```text
1. 实现 Tool Protocol。
2. 定义 name、description、input_schema、aliases。
3. 标注 is_read_only、is_concurrency_safe、is_destructive。
4. 实现 is_enabled(ctx)。
5. 实现 validate_input(tool_input, ctx)。
6. 实现 to_model_spec()。
7. 实现 run(tool_input, ctx) -> ToolResult。
8. 把工具加入 ToolRegistry 构造路径，通常是 BUILTIN_TOOLS。
9. 为权限策略和执行结果补测试。
10. 如有敏感路径或高风险副作用，更新 agent/settings.json。
```
