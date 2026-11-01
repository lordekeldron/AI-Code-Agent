"""Provedores de modelo.

O ``Agent`` fala o formato de mensagens da Anthropic. Cada provedor expõe um objeto
com ``messages.create(...)`` que devolve uma resposta nesse mesmo formato, então o
loop do agente não precisa saber qual modelo está por trás.

- ``anthropic``: API da Anthropic (exige ANTHROPIC_API_KEY e o pacote ``anthropic``).
- ``ollama``: modelo local via Ollama (sem chave, sem conta, sem custo por uso).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# O padrão do Ollama (4096) é pequeno demais para um agente que lê arquivos.
DEFAULT_OLLAMA_NUM_CTX = 16384
OLLAMA_REQUEST_TIMEOUT = 900  # modelos locais podem ser lentos

PROVIDERS = ("anthropic", "ollama")


class ProviderError(RuntimeError):
    """Falha ao falar com o provedor (mensagem pronta para mostrar ao usuário)."""


def default_provider() -> str:
    """Anthropic se houver chave; caso contrário, o modelo local (Ollama)."""
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"


def default_model(provider: str) -> str:
    env = os.environ.get("AI_CODE_AGENT_MODEL")
    if env:
        return env
    return DEFAULT_OLLAMA_MODEL if provider == "ollama" else DEFAULT_ANTHROPIC_MODEL


def make_client(provider: str, host: str | None = None, num_ctx: int | None = None) -> Any:
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "O pacote 'anthropic' não está instalado. Rode: pip install anthropic"
            ) from exc
        return anthropic.Anthropic()
    if provider == "ollama":
        return OllamaClient(host=host, num_ctx=num_ctx)
    raise ProviderError(f"Provedor desconhecido: {provider}")


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(
        block.get("text", "") for block in (system or []) if isinstance(block, dict)
    )


def to_ollama_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converte o histórico (formato Anthropic) para o formato de chat do Ollama."""
    out: list[dict[str, Any]] = []
    system_text = _system_text(system)
    if system_text:
        out.append({"role": "system", "content": system_text})
    tool_names: dict[str, str] = {}  # tool_use_id -> nome, para rotular os resultados

    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        if message["role"] == "assistant":
            texts = [b["text"] for b in content if b["type"] == "text"]
            calls = []
            for block in content:
                if block["type"] == "tool_use":
                    tool_names[block["id"]] = block["name"]
                    calls.append(
                        {"function": {"name": block["name"], "arguments": block["input"]}}
                    )
            item: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(texts)}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
        else:
            for block in content:
                if block["type"] == "tool_result":
                    text = block["content"]
                    if block.get("is_error"):
                        text = f"ERROR: {text}"
                    out.append(
                        {
                            "role": "tool",
                            "tool_name": tool_names.get(block["tool_use_id"], ""),
                            "content": text,
                        }
                    )
                elif block["type"] == "text":
                    out.append({"role": "user", "content": block["text"]})
    return out


def to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def from_ollama_response(data: dict[str, Any]) -> SimpleNamespace:
    """Converte a resposta do Ollama para o formato de blocos da Anthropic."""
    message = data.get("message") or {}
    blocks: list[SimpleNamespace] = []
    text = _THINK_RE.sub("", message.get("content") or "").strip()
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    calls = message.get("tool_calls") or []
    for number, call in enumerate(calls, 1):
        function = call.get("function") or {}
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=call.get("id") or f"call_{number}_{abs(hash(json.dumps(function, sort_keys=True, default=str))) % 10**8}",
                name=function.get("name", ""),
                input=_parse_arguments(function.get("arguments")),
            )
        )
    if calls:
        stop_reason = "tool_use"
    elif data.get("done_reason") == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class _OllamaMessages:
    def __init__(self, client: "OllamaClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: Any,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> SimpleNamespace:
        payload = {
            "model": model,
            "messages": to_ollama_messages(system, messages),
            "tools": to_ollama_tools(tools),
            "stream": False,
            "options": {"num_predict": max_tokens, "num_ctx": self._client.num_ctx},
        }
        return from_ollama_response(self._client.post("/api/chat", payload))


class OllamaClient:
    """Cliente mínimo (só biblioteca padrão) para um servidor Ollama."""

    def __init__(self, host: str | None = None, num_ctx: int | None = None) -> None:
        raw = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        if "://" not in raw:
            raw = f"http://{raw}"
        self.host = raw.rstrip("/")
        self.num_ctx = num_ctx or DEFAULT_OLLAMA_NUM_CTX
        self.messages = _OllamaMessages(self)

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.host + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise ProviderError(f"Ollama respondeu com erro {exc.code}: {detail}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise ProviderError(
                f"Não consegui falar com o Ollama em {self.host} ({exc}). "
                "Ele está instalado e rodando? Veja https://ollama.com/download"
            ) from None
        except json.JSONDecodeError:
            raise ProviderError(f"Resposta inválida do servidor em {self.host}.") from None

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(path, payload, OLLAMA_REQUEST_TIMEOUT)

    def list_models(self) -> list[str]:
        """Nomes dos modelos já baixados; também serve para testar a conexão."""
        data = self._request("/api/tags", None, 10)
        return [m.get("name", "") for m in data.get("models", [])]

    def has_model(self, model: str) -> bool:
        names = self.list_models()
        wanted = model if ":" in model else f"{model}:latest"
        return wanted in names
