#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from semantha_core import DEFAULT_PROTOCOL_VERSION, SemanthaError, SemanthaWorkspace


SERVER_VERSION = "0.1.0"


class JsonRpcError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class SemanthaServer:
    def __init__(self, generator_dir: Path):
        self.workspace = SemanthaWorkspace(generator_dir)
        self.client_capabilities: dict[str, Any] = {}
        self.initialized = False

    def run(self) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send_error(None, -32700, f"Parse error: {exc.msg}")
                continue

            try:
                self._handle_message(message)
            except BrokenPipeError:
                return 0
            except Exception as exc:  # pragma: no cover - safety net
                self._log(traceback.format_exc())
                request_id = message.get("id") if isinstance(message, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {exc}")
        return 0

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "method" in message:
            method = message["method"]
            if "id" in message:
                self._handle_request(message["id"], method, message.get("params") or {})
            else:
                self._handle_notification(method, message.get("params") or {})
            return
        # Server-initiated requests are not used in v0, so stray responses are ignored.

    def _handle_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        try:
            handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
                "initialize": self._handle_initialize,
                "ping": lambda _params: {},
                "tools/list": lambda _params: {"tools": self._tool_definitions()},
                "tools/call": self._handle_tools_call,
                "resources/list": lambda _params: {
                    "resources": self.workspace.list_resources()
                },
                "resources/templates/list": lambda _params: {
                    "resourceTemplates": self.workspace.list_resource_templates()
                },
                "resources/read": self._handle_resources_read,
                "prompts/list": lambda _params: {
                    "prompts": self.workspace.list_prompts()
                },
                "prompts/get": self._handle_prompts_get,
            }
            handler = handlers.get(method)
            if handler is None:
                raise JsonRpcError(-32601, f"Method not found: {method}")
            result = handler(params)
        except JsonRpcError as exc:
            self._send_error(request_id, exc.code, exc.message, exc.data)
            return
        self._send_result(request_id, result)

    def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "notifications/initialized":
            self.initialized = True
            return
        if method.startswith("notifications/"):
            return
        self._log(f"Ignoring notification: {method} {params}")

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested_version = str(
            params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        )
        self.client_capabilities = params.get("capabilities") or {}
        agreed_version = (
            requested_version
            if requested_version == DEFAULT_PROTOCOL_VERSION
            else DEFAULT_PROTOCOL_VERSION
        )
        return {
            "protocolVersion": agreed_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": "SEmantha",
                "title": "SEmantha Semantic Resume MCP Server",
                "version": SERVER_VERSION,
            },
            "instructions": (
                "SEmantha wraps the narrowed markdown-driven resume pipeline. Use retrieval tools to "
                "build a broad selected bundle, let the LLM choose and refine the final subset, then "
                "package it as a resume plan, render LaTeX, and compile PDFs."
            ),
        }

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "").strip()
        arguments = params.get("arguments") or {}
        if not name:
            raise JsonRpcError(-32602, "Missing tool name")
        if not isinstance(arguments, dict):
            raise JsonRpcError(-32602, "Tool arguments must be an object")

        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "semantic_search_projects": lambda args: (
                self.workspace.semantic_search_projects(
                    query=_opt_str(args, "query"),
                    target_text=_opt_str(args, "target_text"),
                    target_file=_opt_str(args, "target_file"),
                    role_family=_opt_str(args, "role_family"),
                    top=_opt_int(args, "top", 10, min_value=1),
                )
            ),
            "inspect_project": lambda args: self.workspace.inspect_project(
                _required_str(args, "project_id")
            ),
            "build_resume_bundle": lambda args: self.workspace.build_resume_bundle(
                query=_opt_str(args, "query"),
                target_text=_opt_str(args, "target_text"),
                target_file=_opt_str(args, "target_file"),
                role_family=_opt_str(args, "role_family"),
                top=_opt_int(args, "top", 6, min_value=1),
                allow_family_duplicates=_opt_bool(
                    args, "allow_family_duplicates", False
                ),
                label=_opt_str(args, "label") or "resume",
            ),
            "create_resume_plan": lambda args: self.workspace.create_resume_plan(
                label=_opt_str(args, "label"),
                selected_file=_opt_str(args, "selected_file"),
                chosen_project_ids=_opt_str_list(args, "chosen_project_ids"),
                project_overrides=_opt_object_of_objects(args, "project_overrides"),
                top_n=_opt_int(args, "top_n", 4, min_value=1),
            ),
            "render_resume_tex": lambda args: self.workspace.render_resume_tex(
                label=_opt_str(args, "label"),
                resume_plan_file=_opt_str(args, "resume_plan_file"),
                output_tex=_opt_str(args, "output_tex"),
                max_projects=_opt_int(args, "max_projects", 4, min_value=1),
                max_bullets_per_project=_opt_int(
                    args, "max_bullets_per_project", 3, min_value=0
                ),
            ),
            "compile_resume_pdf": lambda args: self.workspace.compile_resume_pdf(
                label=_opt_str(args, "label"),
                tex_file=_opt_str(args, "tex_file"),
            ),
        }
        handler = tool_handlers.get(name)
        if handler is None:
            raise JsonRpcError(-32602, f"Unknown tool: {name}")

        try:
            data = handler(arguments)
            return _tool_success(data)
        except JsonRpcError:
            raise
        except SemanthaError as exc:
            return _tool_failure(str(exc))
        except Exception as exc:  # pragma: no cover - safety net
            self._log(traceback.format_exc())
            return _tool_failure(f"Unhandled tool error: {exc}")

    def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = _required_str(params, "uri")
        try:
            content = self.workspace.read_resource(uri)
        except SemanthaError as exc:
            raise JsonRpcError(-32002, str(exc)) from exc
        return {"contents": [content]}

    def _handle_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _required_str(params, "name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(-32602, "Prompt arguments must be an object")
        try:
            return self.workspace.get_prompt(name, arguments)
        except SemanthaError as exc:
            raise JsonRpcError(-32602, str(exc)) from exc

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "semantic_search_projects",
                "title": "Semantic Search Projects",
                "description": "Rank project records against a query, target text, or target file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "target_text": {"type": "string"},
                        "target_file": {"type": "string"},
                        "role_family": {"type": "string"},
                        "top": {"type": "integer", "minimum": 1, "default": 10},
                    },
                },
            },
            {
                "name": "inspect_project",
                "title": "Inspect Project",
                "description": "Return the full project record, context overlay, and family metadata for one project_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"project_id": {"type": "string"}},
                    "required": ["project_id"],
                },
            },
            {
                "name": "build_resume_bundle",
                "title": "Build Resume Bundle",
                "description": "Rank projects for a target, write the selected bundle JSON, and emit an LLM-ready prompt bundle.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "target_text": {"type": "string"},
                        "target_file": {"type": "string"},
                        "role_family": {"type": "string"},
                        "top": {"type": "integer", "minimum": 1, "default": 6},
                        "allow_family_duplicates": {
                            "type": "boolean",
                            "default": False,
                        },
                        "label": {"type": "string"},
                    },
                },
            },
            {
                "name": "create_resume_plan",
                "title": "Create Resume Plan",
                "description": "Create the ordered resume-plan editorial intermediary from a selected bundle, with optional subset selection and per-project overrides.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "selected_file": {"type": "string"},
                        "chosen_project_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "project_overrides": {"type": "object"},
                        "top_n": {"type": "integer", "minimum": 1, "default": 4},
                    },
                },
            },
            {
                "name": "render_resume_tex",
                "title": "Render Resume TeX",
                "description": "Render deterministic LaTeX from a resume plan or selected bundle using the current profile and template.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "resume_plan_file": {"type": "string"},
                        "output_tex": {"type": "string"},
                        "max_projects": {"type": "integer", "minimum": 1, "default": 4},
                        "max_bullets_per_project": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 3,
                        },
                    },
                },
            },
            {
                "name": "compile_resume_pdf",
                "title": "Compile Resume PDF",
                "description": "Compile a rendered TeX resume into PDF with pdflatex and report warnings/page count.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "tex_file": {"type": "string"},
                    },
                },
            },
        ]

    def _send_result(self, request_id: Any, result: Any) -> None:
        self._write_message({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _send_error(
        self, request_id: Any, code: int, message: str, data: Any | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            payload["error"]["data"] = data
        self._write_message(payload)

    def _write_message(self, payload: dict[str, Any]) -> None:
        sys.stdout.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()

    def _log(self, message: str) -> None:
        sys.stderr.write(f"[SEmantha] {message}\n")
        sys.stderr.flush()


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(-32602, f"{key} must be a non-empty string")
    return value.strip()


def _opt_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise JsonRpcError(-32602, f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def _opt_int(
    arguments: dict[str, Any], key: str, default: int, min_value: int | None = None
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonRpcError(-32602, f"{key} must be an integer")
    if min_value is not None and value < min_value:
        raise JsonRpcError(-32602, f"{key} must be >= {min_value}")
    return value


def _opt_bool(arguments: dict[str, Any], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise JsonRpcError(-32602, f"{key} must be a boolean")
    return value


def _opt_str_list(arguments: dict[str, Any], key: str) -> list[str] | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise JsonRpcError(-32602, f"{key} must be an array of strings")
    return value


def _opt_object_of_objects(
    arguments: dict[str, Any], key: str
) -> dict[str, dict[str, Any]] | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(map_key, str) or not isinstance(map_value, dict)
        for map_key, map_value in value.items()
    ):
        raise JsonRpcError(-32602, f"{key} must be an object keyed by strings")
    return value


def _required_object(arguments: dict[str, Any], key: str) -> dict[str, Any]:
    value = arguments.get(key)
    if not isinstance(value, dict):
        raise JsonRpcError(-32602, f"{key} must be an object")
    return value


def _tool_success(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
        "isError": False,
    }


def _tool_failure(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def main() -> int:
    server = SemanthaServer(Path(__file__).resolve().parent)
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
