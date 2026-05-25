from pathlib import Path


def resolve_workspace_path(
    workspace_root: Path,
    input_path: str,
) -> Path:
    """
    将模型传入路径解析到 workspace 内部。

    禁止路径逃逸。
    """
    root = workspace_root.resolve()

    raw_path = Path(input_path)

    if raw_path.is_absolute():
        target = raw_path.resolve()
    else:
        target = (root / raw_path).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Path outside workspace: {input_path}")

    return target