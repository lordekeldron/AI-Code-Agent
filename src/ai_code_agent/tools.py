"""Ferramentas que o agente usa para trabalhar dentro de um workspace.

Cada ferramenta recebe o ``Workspace`` e os argumentos escolhidos pelo modelo e
devolve uma string. Erros previstos levantam ``ToolError``; a mensagem volta ao
modelo como resultado de ferramenta com ``is_error=True`` para ele se corrigir.

Segurança: o confinamento ao workspace vale para as ferramentas de arquivo.
``run_command`` executa um shell de verdade e NÃO é um sandbox. A proteção real
vem da confirmação do usuário antes de cada execução; a lista de comandos
bloqueados é apenas uma rede de segurança de melhor esforço.
"""

from __future__ import annotations

import fnmatch
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

MAX_OUTPUT_CHARS = 20_000
MAX_READ_LINES = 2_000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_COMMAND_TIMEOUT = 600

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)

_BLOCKED_COMMANDS = [
    re.compile(pattern)
    for pattern in (
        # rm recursivo apontando para a raiz, o home ou um curinga na raiz
        r"\brm\s+(-\w+\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(/|/\*|~|~/|\$HOME|\$HOME/)(\s|$)",
        r"\bmkfs(\.\w+)?\b",
        r":\(\)\s*\{.*\|.*&.*\}\s*;",  # fork bomb
        r"\bdd\b[^|;&]*\bof=/dev/",
        r">\s*/dev/(sd|nvme|hd)",
    )
]


class ToolError(Exception):
    """Erro esperado de uma ferramenta; a mensagem é devolvida ao modelo."""


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Corta textos longos mantendo o começo e o fim."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def is_blocked_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in _BLOCKED_COMMANDS)


class Workspace:
    """Diretório raiz em que o agente pode ler e escrever arquivos."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.root}")

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` (relativo à raiz) e recusa o que escapar do workspace."""
        if not isinstance(path, str) or not path:
            raise ToolError("path must be a non-empty string")
        candidate = (self.root / path).resolve()  # segue symlinks
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"Path is outside the workspace: {path}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def _read_text(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return fh.read()
    except UnicodeDecodeError:
        raise ToolError(f"{path.name} is not a UTF-8 text file") from None
    except OSError as exc:
        raise ToolError(f"Could not read file: {exc}") from None


def _write_text(path: Path, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    except OSError as exc:
        raise ToolError(f"Could not write file: {exc}") from None


# --------------------------------------------------------------------------- #
# Implementações
# --------------------------------------------------------------------------- #


def list_dir(ws: Workspace, path: str = ".", max_depth: int = 1) -> str:
    base = ws.resolve(path)
    if not base.is_dir():
        raise ToolError(f"Not a directory: {path}")
    depth_limit = max(1, min(int(max_depth), 5))
    lines: list[str] = []
    truncated = False

    def walk(directory: Path, depth: int) -> None:
        nonlocal truncated
        try:
            entries = sorted(
                directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
            )
        except OSError:
            return
        for entry in entries:
            if len(lines) >= MAX_LIST_ENTRIES:
                truncated = True
                return
            indent = "  " * (depth - 1)
            if entry.is_symlink():
                lines.append(f"{indent}{entry.name}@")
            elif entry.is_dir():
                if entry.name in IGNORED_DIRS:
                    continue
                lines.append(f"{indent}{entry.name}/")
                if depth < depth_limit:
                    walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{entry.name}")

    walk(base, 1)
    header = f"{ws.relative(base)}/"
    if not lines:
        return f"{header} (empty)"
    suffix = f"\n... (truncated at {MAX_LIST_ENTRIES} entries)" if truncated else ""
    return header + "\n" + "\n".join(lines) + suffix


def read_file(
    ws: Workspace, path: str, start_line: int = 1, end_line: int | None = None
) -> str:
    file = ws.resolve(path)
    if not file.is_file():
        raise ToolError(f"Not a file: {path}")
    lines = _read_text(file).splitlines()
    total = len(lines)
    rel = ws.relative(file)
    if total == 0:
        return f"{rel} (empty file)"
    start = max(1, int(start_line))
    if start > total:
        raise ToolError(f"start_line {start} is past the end of the file ({total} lines)")
    wanted_end = int(end_line) if end_line is not None else total
    end = min(total, wanted_end, start + MAX_READ_LINES - 1)
    if end < start:
        raise ToolError("end_line must be >= start_line")
    body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(start, end + 1))
    note = (
        f"\n[more lines follow; call again with start_line={end + 1}]"
        if end < total
        else ""
    )
    return truncate(f"{rel} (lines {start}-{end} of {total})\n{body}") + note


def write_file(ws: Workspace, path: str, content: str) -> str:
    file = ws.resolve(path)
    if file.is_dir():
        raise ToolError(f"{path} is a directory")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    existed = file.exists()
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"Could not create directories: {exc}") from None
    _write_text(file, content)
    verb = "Overwrote" if existed else "Created"
    size = len(content.encode("utf-8"))
    return f"{verb} {ws.relative(file)} ({size} bytes, {len(content.splitlines())} lines)"


def edit_file(
    ws: Workspace,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    file = ws.resolve(path)
    if not file.is_file():
        raise ToolError(f"Not a file: {path}")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        raise ToolError("old_string and new_string must be strings")
    if old_string == "":
        raise ToolError(
            "old_string must not be empty (use write_file to create or replace a whole file)"
        )
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical; nothing to change")
    text = _read_text(file)
    count = text.count(old_string)
    if count == 0:
        raise ToolError(
            "old_string not found. Read the file again and copy the text exactly, "
            "including whitespace."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {count} times; add surrounding context to make it "
            "unique or set replace_all=true"
        )
    if replace_all:
        updated = text.replace(old_string, new_string)
    else:
        updated = text.replace(old_string, new_string, 1)
    _write_text(file, updated)
    done = count if replace_all else 1
    plural = "s" if done != 1 else ""
    return f"Edited {ws.relative(file)} ({done} replacement{plural})"


def _iter_files(base: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            file = Path(dirpath) / name
            if file.is_symlink():
                continue
            yield file


def search(
    ws: Workspace,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    max_results: int = 100,
) -> str:
    base = ws.resolve(path)
    if not base.exists():
        raise ToolError(f"Path does not exist: {path}")
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"Invalid regex: {exc}") from None
    limit = max(1, min(int(max_results), 500))
    files = [base] if base.is_file() else _iter_files(base)
    matches: list[str] = []
    for file in files:
        rel = ws.relative(file)
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(file.name, glob)):
            continue
        try:
            if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            with open(file, encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    if regex.search(line):
                        matches.append(f"{rel}:{number}: {line.rstrip()[:300]}")
                        if len(matches) >= limit:
                            break
        except (UnicodeDecodeError, OSError):
            continue
        if len(matches) >= limit:
            break
    if not matches:
        return "No matches."
    note = f"\n[stopped at {limit} results; refine the pattern]" if len(matches) >= limit else ""
    return truncate("\n".join(matches)) + note


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_command(ws: Workspace, command: str, timeout: int = 60) -> str:
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty string")
    if is_blocked_command(command):
        raise ToolError(
            "Command blocked by the safety policy (destructive system-level operation)."
        )
    seconds = max(1, min(int(timeout), MAX_COMMAND_TIMEOUT))
    popen_kwargs: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {}
    # A chave da API não deve vazar para os comandos executados pelo agente.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=ws.root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except OSError as exc:
        raise ToolError(f"Could not start command: {exc}") from None
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = ""
    except BaseException:
        _kill_process_tree(proc)
        raise
    if timed_out:
        status = f"[timed out after {seconds}s; process killed]"
    else:
        status = f"[exit code {proc.returncode}]"
    return truncate(f"{status}\n{output or ''}")


# --------------------------------------------------------------------------- #
# Registro das ferramentas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., str]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        """Definição no formato esperado pela API de mensagens."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def build_tools() -> dict[str, Tool]:
    tools = [
        Tool(
            name="list_dir",
            description=(
                "List files and directories inside the workspace. Directories end with '/'. "
                "Noise such as .git and node_modules is hidden. Use max_depth to look deeper."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the workspace root. Default '.'.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "How many levels to show (1-5). Default 1.",
                    },
                },
                "required": [],
            },
            func=list_dir,
        ),
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file. Output has line numbers (not part of the file). "
                "Large files are returned in chunks; use start_line/end_line to page."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "start_line": {"type": "integer", "description": "First line to read (1-based). Default 1."},
                    "end_line": {"type": "integer", "description": "Last line to read, inclusive. Default: end of file."},
                },
                "required": ["path"],
            },
            func=read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Create a new file or completely overwrite an existing one. Parent directories "
                "are created automatically. For small changes to an existing file prefer edit_file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "content": {"type": "string", "description": "Complete file content."},
                },
                "required": ["path", "content"],
            },
            func=write_file,
            needs_approval=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact piece of text in an existing file. old_string must match the "
                "file exactly (including whitespace and indentation) and be unique unless "
                "replace_all is true. Read the file first."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring a unique match.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            func=edit_file,
            needs_approval=True,
        ),
        Tool(
            name="search",
            description=(
                "Search file contents with a regular expression (Python syntax). Returns "
                "'path:line: text' matches. Skips binary files and noise directories."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to look for."},
                    "path": {"type": "string", "description": "File or directory to search. Default '.'."},
                    "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive match."},
                    "max_results": {"type": "integer", "description": "Maximum matches to return (1-500). Default 100."},
                },
                "required": ["pattern"],
            },
            func=search,
        ),
        Tool(
            name="run_command",
            description=(
                "Run a shell command in the workspace root and return its combined stdout/stderr "
                "and exit code. Use it to run tests, linters, builds and git. There is no stdin; "
                "commands must not wait for input. Long-running commands are killed at the timeout."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {
                        "type": "integer",
                        "description": f"Timeout in seconds (1-{MAX_COMMAND_TIMEOUT}). Default 60.",
                    },
                },
                "required": ["command"],
            },
            func=run_command,
            needs_approval=True,
        ),
    ]
    return {tool.name: tool for tool in tools}


def execute_tool(ws: Workspace, tool: Tool, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Executa uma ferramenta e devolve ``(texto, is_error)`` sem nunca levantar."""
    try:
        return tool.func(ws, **arguments), False
    except ToolError as exc:
        return str(exc), True
    except (TypeError, ValueError) as exc:
        return f"Invalid arguments for {tool.name}: {exc}", True
    except Exception as exc:  # noqa: BLE001 - o modelo precisa ver qualquer falha
        return f"Unexpected error in {tool.name}: {type(exc).__name__}: {exc}", True
