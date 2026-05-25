from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolContext:
    """工具执行时的运行时上下文。

    这不是模型输入，而是 agent runtime 注入给工具的环境信息。
    """

    workspace_root: Path
    permission_mode: str = "default"
    session_id: str | None = None
    agent_id: str | None = None
    skill_call_depth: int = 0
    active_skill_name: str | None = None
    team_name: str | None = None
    agent_name: str | None = None
    subagent_depth: int = 0


@dataclass(frozen=True)
class ToolCall:
    """内部 tool_call 格式。

    LangGraph / LLM 原始 tool call 会先由 adapter 转成这个结构。
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """内部工具执行结果格式。

    执行完成后，再由 adapter 转成 LangGraph tool message。
    """

    ok: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None


class ToolInputError(Exception):
    pass


class ToolExecutionError(Exception):
    pass


@runtime_checkable
class Tool(Protocol):
    """CC-style Tool 协议。

    注意：
    1. 这是 Protocol，不是 ABC。
    2. 具体工具不需要显式继承 Tool。
    3. 只要对象拥有这些字段和方法，就可以被当作 Tool 使用。
    """

    # 工具的唯一主名称。模型产生 tool_call 时会用这个名字请求工具，
    # ToolRegistry 也会用它建立主索引。
    name: str

    # 面向模型和开发者的简短能力描述。它会进入模型可见的 tool spec，
    # 影响模型什么时候选择调用该工具。
    description: str

    # 工具输入的 JSON Schema。当前 executor 至少会检查 required 字段，
    # 未来也可以替换成更完整的 jsonschema/pydantic 校验。
    input_schema: dict[str, Any]

    # 工具别名列表。Registry 会把 alias 注册到同一个工具对象，
    # 用于兼容历史名称或更自然的模型调用名称。
    aliases: list[str]

    # 是否只读。只读工具理论上不修改 workspace 或外部状态，
    # 后续 permission gate 和并发调度会依赖这个标记。
    is_read_only: bool

    # 是否并发安全。为 True 的工具可以被未来的 orchestrator 并发执行；
    # 写文件、shell 等有副作用的工具通常应为 False。
    is_concurrency_safe: bool

    # 是否具有破坏性。删除、覆盖、执行命令等工具应标记为 True，
    # 方便后续接入人工审批或更严格的权限策略。
    is_destructive: bool

    def is_enabled(self, ctx: ToolContext) -> bool:
        """判断工具在当前运行上下文中是否可用。

        这里可以根据 workspace、session、agent_id、feature flag 或权限配置
        动态禁用工具。Registry 查找和暴露工具列表时会调用它。
        """
        ...

    def validate_input(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        """执行工具自己的语义校验。

        schema 校验只能保证字段存在和基本结构；这里负责更具体的约束，
        例如 path 必须是非空字符串、overwrite 必须是 bool、路径不能逃逸
        workspace 等。校验失败时抛出异常，由 ToolExecutor 归一化为
        invalid_tool_input。
        """
        ...

    def to_model_spec(self) -> dict[str, Any]:
        """生成模型可见的工具描述。

        返回值应该只包含模型需要知道的信息，例如 name、description、
        input_schema。不要把 ToolContext、内部实现、权限细节或文件系统对象
        暴露给模型。
        """
        ...

    def run(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        """执行工具主体逻辑。

        所有具体工具副作用都应该封装在这里，例如读文件、写文件或未来执行
        shell。Graph 层不能绕过 ToolExecutor 直接调用这些逻辑。返回值必须是
        ToolResult，由 adapter 再转换成模型可见的 ToolMessage。
        """
        ...
