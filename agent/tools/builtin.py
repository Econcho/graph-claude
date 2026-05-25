from .base import Tool
from .command_tools import RunCommandTool, RunTestsTool
from .diagnostic_tools import CollectDiagnosticsTool, ParseTestOutputTool
from .edit_tools import EditFileTool
from .file_tools import (
    DeleteFileTool,
    MkdirTool,
    MoveFileTool,
    ReadFileTool,
    ReadManyFilesTool,
    WriteFileTool,
)
from .git_tools import GitDiffTool, GitLogTool, GitShowTool, GitStatusTool
from .search_tools import GlobTool, GrepTool, ListDirTool
from .symbol_tools import FindDefinitionTool, FindReferencesTool, OutlineFileTool


BUILTIN_TOOLS: list[Tool] = [
    ReadFileTool(),
    ReadManyFilesTool(),
    WriteFileTool(),
    ListDirTool(),
    GlobTool(),
    GrepTool(),
    EditFileTool(),
    RunCommandTool(),
    RunTestsTool(),
    MkdirTool(),
    MoveFileTool(),
    DeleteFileTool(),
    GitStatusTool(),
    GitDiffTool(),
    GitLogTool(),
    GitShowTool(),
    ParseTestOutputTool(),
    CollectDiagnosticsTool(),
    OutlineFileTool(),
    FindDefinitionTool(),
    FindReferencesTool(),
]
