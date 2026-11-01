"""Interface de linha de comando do AI Code Agent."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .agent import Agent, AgentConfig, Hooks
from .providers import (
    DEFAULT_OLLAMA_MODEL,
    PROVIDERS,
    OllamaClient,
    ProviderError,
    default_model,
    default_provider,
    make_client,
)
from .tools import Workspace

PREVIEW_LIMIT = 600

BANNER = "AI Code Agent — descreva a tarefa. Comandos: /ajuda, /reset, /sair"
HELP = """Comandos:
  /reset   começa uma conversa nova (esquece o histórico)
  /sair    encerra (também: /exit, Ctrl+D)
  /ajuda   mostra esta mensagem

Edições de arquivos e comandos pedem sua confirmação antes de rodar:
  s = sim, n = não (padrão), t = sim para todas nesta sessão."""


def _clip(text: Any, limit: int = PREVIEW_LIMIT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} caracteres]"


def short_call(name: str, args: dict[str, Any]) -> str:
    if name == "run_command":
        return f"$ {_clip(args.get('command', ''), 120)}"
    target = args.get("path") or args.get("pattern") or ""
    return f"{name} {_clip(target, 100)}".rstrip()


def describe_call(name: str, args: dict[str, Any]) -> str:
    """Texto mostrado ao usuário quando uma ação pede aprovação."""
    if name == "run_command":
        return (
            f"Executar comando (timeout {args.get('timeout', 60)}s):\n"
            f"  $ {args.get('command', '')}"
        )
    if name == "write_file":
        content = str(args.get("content", ""))
        return (
            f"Escrever o arquivo {args.get('path')} ({len(content)} caracteres):\n"
            f"{_clip(content)}"
        )
    if name == "edit_file":
        return (
            f"Editar {args.get('path')}:\n"
            f"--- remover\n{_clip(args.get('old_string', ''))}\n"
            f"+++ inserir\n{_clip(args.get('new_string', ''))}"
        )
    return short_call(name, args)


class Approver:
    """Pergunta no terminal se uma ação pode rodar."""

    def __init__(self) -> None:
        self.allow_all = False

    def __call__(self, name: str, args: dict[str, Any]) -> bool:
        if self.allow_all:
            return True
        print(f"\n⚠ Pedido de permissão\n{describe_call(name, args)}")
        while True:
            try:
                answer = input("Permitir? [s]im / [n]ão / [t]odas nesta sessão: ").strip().lower()
            except EOFError:
                return False
            if answer in {"s", "sim", "y", "yes"}:
                return True
            if answer in {"", "n", "não", "nao", "no"}:
                return False
            if answer in {"t", "todas", "a", "all"}:
                self.allow_all = True
                return True


def _print_text(text: str) -> None:
    print(f"\n{text}")


def _print_tool_use(name: str, args: dict[str, Any]) -> None:
    print(f"\n→ {short_call(name, args)}")


def _print_tool_result(_name: str, text: str, is_error: bool) -> None:
    first = next((line for line in text.splitlines() if line.strip()), "")
    mark = "✗" if is_error else "✓"
    print(f"  {mark} {_clip(first, 120)}")


def _print_notice(message: str) -> None:
    print(f"\n[aviso] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-code-agent",
        description="Agente de IA que lê, edita e testa código no diretório de trabalho.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Tarefa a executar. Sem isso, abre o modo interativo.",
    )
    parser.add_argument(
        "-w", "--workspace", default=".", help="Diretório do projeto (padrão: o atual)."
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="Modelo por trás do agente. Padrão: 'anthropic' se ANTHROPIC_API_KEY "
        "estiver definida; senão 'ollama' (modelo local, sem chave).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Modelo a usar (padrão: {DEFAULT_OLLAMA_MODEL} no ollama, claude-sonnet-5-5 "
        "na anthropic; ou defina AI_CODE_AGENT_MODEL).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Endereço do servidor Ollama (padrão: OLLAMA_HOST ou http://localhost:11434).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Janela de contexto pedida ao Ollama, em tokens (padrão: 16384).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Aprova automaticamente edições e comandos (use com cuidado).",
    )
    parser.add_argument("--max-turns", type=int, default=40, help="Máximo de passos por tarefa.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Máximo de tokens por resposta.")
    return parser


def _run_safely(agent: Agent, prompt: str) -> int:
    try:
        agent.run(prompt)
        return 0
    except KeyboardInterrupt:
        print("\n[interrompido] a mensagem foi descartada; envie de novo se quiser continuar.")
        return 130
    except Exception as exc:  # noqa: BLE001 - mostrar o erro em vez de derrubar o REPL
        print(f"\n[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _repl(agent: Agent) -> int:
    print(BANNER)
    while True:
        try:
            line = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/sair", "/exit", "/quit"}:
            return 0
        if line == "/reset":
            agent.reset()
            print("Conversa reiniciada.")
            continue
        if line == "/ajuda":
            print(HELP)
            continue
        _run_safely(agent, line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    provider = args.provider or default_provider()
    model = args.model or default_model(provider)
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Erro: defina ANTHROPIC_API_KEY ou use o modelo local sem chave: "
            "ai-code-agent --provider ollama",
            file=sys.stderr,
        )
        return 2
    try:
        workspace = Workspace(args.workspace)
        client = make_client(provider, host=args.host, num_ctx=args.num_ctx)
        if isinstance(client, OllamaClient) and not client.has_model(model):
            print(
                f"Erro: o modelo '{model}' ainda não foi baixado no Ollama.\n"
                f"Rode primeiro: ollama pull {model}",
                file=sys.stderr,
            )
            return 2
    except (ValueError, ProviderError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2

    agent = Agent(
        workspace,
        client=client,
        config=AgentConfig(
            model=model,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            auto_approve=args.yes,
        ),
        hooks=Hooks(
            on_text=_print_text,
            on_tool_use=_print_tool_use,
            on_tool_result=_print_tool_result,
            on_notice=_print_notice,
            approve=Approver(),
        ),
    )
    print(f"Workspace: {workspace.root}\nModelo: {model} ({provider})")
    if args.yes:
        print("[aviso] aprovação automática ligada: edições e comandos rodam sem perguntar.")

    if args.prompt:
        return _run_safely(agent, " ".join(args.prompt))
    return _repl(agent)


if __name__ == "__main__":
    raise SystemExit(main())
