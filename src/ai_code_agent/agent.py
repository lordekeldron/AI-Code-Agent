"""O loop agêntico: envia a conversa ao modelo, executa as ferramentas pedidas e repete."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .tools import Tool, Workspace, build_tools, execute_tool

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are AI Code Agent, an autonomous software engineering assistant. You work inside a single project directory (the workspace) on the user's machine and can read, search, create and edit files and run shell commands through the provided tools.

Workspace root: {workspace}

How to work:
- Understand before changing. List the directory and read the relevant files before editing; search the code base instead of guessing where things live.
- Read a file before you edit it, and keep edits minimal and consistent with the existing style. Prefer edit_file for changes to existing files and write_file for new files or full rewrites.
- Verify your work. After changing code, run the project's tests, linter or the program itself with run_command whenever possible, and fix what breaks.
- Never invent file contents, command output or test results. If something fails, read the error and investigate.
- All file paths are relative to the workspace root. You cannot access files outside it.
- Some tools need the user's approval. If the user denies a call, do not retry the same action; explain what you wanted to do and propose an alternative or ask how to proceed.
- Avoid destructive or irreversible actions (deleting many files, force pushes, rewriting history) unless the user explicitly asked for them.
- If the request is ambiguous or risky, ask a short clarifying question instead of guessing.
- Keep working until the task is done, then finish with a short summary of what you changed and anything the user should check. Reply in the same language the user writes in.
"""

DENIED_MESSAGE = (
    "The user denied permission for this action, so it was not executed. "
    "Do not retry it unchanged; explain what you intended and ask how to proceed."
)


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = 8192
    max_turns: int = 40
    auto_approve: bool = False


@dataclass
class Hooks:
    """Callbacks opcionais para a interface acompanhar o agente."""

    on_text: Callable[[str], None] | None = None
    on_tool_use: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_result: Callable[[str, str, bool], None] | None = None
    on_notice: Callable[[str], None] | None = None
    #: Decide se uma ferramenta que exige aprovação pode rodar. Sem ele, tudo é negado.
    approve: Callable[[str, dict[str, Any]], bool] | None = None


def make_client() -> Any:
    """Cria o cliente da Anthropic (lê ANTHROPIC_API_KEY do ambiente)."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "O pacote 'anthropic' não está instalado. Rode: pip install anthropic"
        ) from exc
    return anthropic.Anthropic()


def _block_to_param(block: Any) -> dict[str, Any] | None:
    """Converte um bloco da resposta em um bloco aceito de volta pela API."""
    kind = getattr(block, "type", None)
    if kind == "text":
        text = getattr(block, "text", "")
        # A API rejeita blocos de texto vazios quando a conversa é reenviada.
        return {"type": "text", "text": text} if text.strip() else None
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input or {}),
        }
    dump = getattr(block, "model_dump", None)
    if callable(dump):  # ex.: blocos de thinking precisam voltar intactos
        return dump(exclude_none=True)
    return None


class Agent:
    def __init__(
        self,
        workspace: Workspace,
        client: Any | None = None,
        config: AgentConfig | None = None,
        hooks: Hooks | None = None,
        tools: dict[str, Tool] | None = None,
    ) -> None:
        self.workspace = workspace
        self.client = client if client is not None else make_client()
        self.config = config or AgentConfig()
        self.hooks = hooks or Hooks()
        self.tools = tools if tools is not None else build_tools()
        self.messages: list[dict[str, Any]] = []
        self._tool_schemas = [tool.schema() for tool in self.tools.values()]
        self._system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT.format(workspace=workspace.root),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def reset(self) -> None:
        """Começa uma conversa nova."""
        self.messages.clear()

    def run(self, user_message: str) -> str:
        """Processa uma mensagem do usuário até o modelo concluir e devolve o texto final.

        Se algo falhar no meio (erro de API, Ctrl+C), a conversa volta ao estado anterior
        para nunca ficar com um ``tool_use`` sem o ``tool_result`` correspondente.
        """
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_message})
        try:
            return self._loop()
        except BaseException:
            del self.messages[checkpoint:]
            raise

    # ------------------------------------------------------------------ #

    def _loop(self) -> str:
        final_text = ""
        for _ in range(self.config.max_turns):
            response = self._create_message()
            blocks = [
                param
                for param in (_block_to_param(b) for b in response.content)
                if param is not None
            ]
            stop_reason = response.stop_reason

            texts = [b["text"] for b in blocks if b["type"] == "text"]
            for text in texts:
                self._call(self.hooks.on_text, text)
            if texts:
                final_text = "\n\n".join(texts)

            if stop_reason != "tool_use":
                # tool_use que não vai ser executado (ex.: cortado por max_tokens)
                # não pode ficar no histórico sem tool_result.
                blocks = [b for b in blocks if b["type"] != "tool_use"] or [
                    {"type": "text", "text": "(no output)"}
                ]
                self.messages.append({"role": "assistant", "content": blocks})
                if stop_reason == "max_tokens":
                    self._call(
                        self.hooks.on_notice,
                        "A resposta foi cortada por atingir o limite de tokens "
                        "(max_tokens). Peça para continuar ou aumente --max-tokens.",
                    )
                elif stop_reason not in ("end_turn", "stop_sequence"):
                    self._call(self.hooks.on_notice, f"Resposta encerrada com motivo: {stop_reason}")
                return final_text

            self.messages.append({"role": "assistant", "content": blocks})
            results = [self._run_tool(b) for b in blocks if b["type"] == "tool_use"]
            if not results:
                return final_text
            self.messages.append({"role": "user", "content": results})

        self._call(
            self.hooks.on_notice,
            f"Parei após {self.config.max_turns} passos sem concluir. "
            "Envie uma mensagem para continuar ou aumente --max-turns.",
        )
        return final_text

    def _create_message(self) -> Any:
        return self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self._system,
            tools=self._tool_schemas,
            messages=self.messages,
        )

    def _run_tool(self, block: dict[str, Any]) -> dict[str, Any]:
        name, args, tool_id = block["name"], block["input"], block["id"]
        self._call(self.hooks.on_tool_use, name, args)
        tool = self.tools.get(name)
        if tool is None:
            text = f"Unknown tool: {name}. Available tools: {', '.join(self.tools)}"
            is_error = True
        elif tool.needs_approval and not self._approved(name, args):
            text, is_error = DENIED_MESSAGE, True
        else:
            text, is_error = execute_tool(self.workspace, tool, args)
        self._call(self.hooks.on_tool_result, name, text, is_error)
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": text or "(no output)",
        }
        if is_error:
            result["is_error"] = True
        return result

    def _approved(self, name: str, args: dict[str, Any]) -> bool:
        if self.config.auto_approve:
            return True
        return self.hooks.approve is not None and bool(self.hooks.approve(name, args))

    @staticmethod
    def _call(fn: Callable[..., None] | None, *args: Any) -> None:
        if fn is not None:
            fn(*args)
