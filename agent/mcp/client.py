from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.mcp.models import McpServerConfig


class McpConnectionError(Exception):
    pass


class McpProtocolError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        data: Any = None,
    ):
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass
class StdioMcpClient:
    config: McpServerConfig
    timeout_seconds: float = 30
    cwd: Path | None = None
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _next_id: int = field(default=1, init=False)
    _responses: dict[int, queue.Queue[dict[str, Any]]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stderr_lines: list[str] = field(default_factory=list, init=False)

    def connect(self) -> None:
        if self.config.type != "stdio":
            raise McpConnectionError(f"unsupported_transport:{self.config.type}")
        if not self.config.command:
            raise McpConnectionError("stdio MCP server requires command")

        env = os.environ.copy()
        env.update(self.config.env)
        argv = [self.config.command, *self.config.args]
        try:
            self._process = subprocess.Popen(
                argv,
                cwd=self.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            raise McpConnectionError(str(error)) from error

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "cc-langgraph",
                    "version": "0.1.0",
                },
            },
        )
        self.notify("notifications/initialized", {})

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        self._process = None

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.request("tools/list", {})
        tools = response.get("tools", [])
        if not isinstance(tools, list):
            raise McpProtocolError("tools/list response must contain tools list")
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._allocate_id()
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._responses[request_id] = response_queue

        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        try:
            response = response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            self._responses.pop(request_id, None)
            raise McpConnectionError(f"MCP request timed out: {method}") from error

        if "error" in response:
            raw_error = response.get("error")
            if isinstance(raw_error, dict):
                raise McpProtocolError(
                    str(raw_error.get("message", "MCP error")),
                    code=raw_error.get("code"),
                    data=raw_error.get("data"),
                )
            raise McpProtocolError(str(raw_error))

        result = response.get("result", {})
        if not isinstance(result, dict):
            raise McpProtocolError("MCP result must be an object")
        return result

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def _allocate_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            return request_id

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpConnectionError("MCP server is not connected")
        if process.poll() is not None:
            stderr = "\n".join(self._stderr_lines[-10:])
            raise McpConnectionError(f"MCP server exited with code {process.returncode}: {stderr}")

        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict) or "id" not in message:
                continue
            try:
                request_id = int(message["id"])
            except (TypeError, ValueError):
                continue
            response_queue = self._responses.pop(request_id, None)
            if response_queue is not None:
                response_queue.put(message)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_lines.append(line.rstrip())
            if len(self._stderr_lines) > 50:
                self._stderr_lines = self._stderr_lines[-50:]
