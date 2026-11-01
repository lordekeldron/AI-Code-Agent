import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_code_agent import Agent, AgentConfig, Workspace
from ai_code_agent.cli import main
from ai_code_agent.providers import (
    OllamaClient,
    ProviderError,
    default_provider,
    from_ollama_response,
    to_ollama_messages,
    to_ollama_tools,
)


class FakeOllama:
    """Servidor HTTP que imita o Ollama e grava o que recebe."""

    def __init__(self, chat_responses, models=("qwen3:8b",)):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if not chat_responses:
                    self._send(500, {"error": "no more scripted responses"})
                else:
                    self._send(200, chat_responses.pop(0))

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_ollama():
    servers = []

    def _make(*args, **kwargs):
        server = FakeOllama(*args, **kwargs)
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.close()


def chat(content="", tool_calls=None, done_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message, "done": True, "done_reason": done_reason}


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


# --- conversão ----------------------------------------------------------------


def test_history_is_converted_to_ollama_format():
    system = [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "leia a.txt"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "lendo"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "conteudo"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "run_command", "input": {"command": "x"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "nope", "is_error": True},
            ],
        },
    ]
    out = to_ollama_messages(system, messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "leia a.txt"}
    assert out[2]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]
    assert out[2]["content"] == "lendo"
    assert out[3] == {"role": "tool", "tool_name": "read_file", "content": "conteudo"}
    assert out[5] == {"role": "tool", "tool_name": "run_command", "content": "ERROR: nope"}


def test_tools_are_converted_to_function_schemas():
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    assert to_ollama_tools(tools) == [
        {"type": "function", "function": {"name": "t", "description": "d",
                                          "parameters": {"type": "object", "properties": {}}}}
    ]


def test_response_conversion_covers_tools_text_and_limits():
    r = from_ollama_response(chat("<think>pensando</think>Vou ler.", [call("read_file", path="a")]))
    assert r.stop_reason == "tool_use"
    assert r.content[0].text == "Vou ler."
    assert r.content[1].type == "tool_use" and r.content[1].input == {"path": "a"}
    assert r.content[1].id  # o Ollama não manda id; geramos um

    # argumentos como string JSON, e ids distintos para chamadas distintas
    r = from_ollama_response(
        chat("", [{"function": {"name": "a", "arguments": '{"x": 1}'}}, call("a", x=2)])
    )
    assert r.content[0].input == {"x": 1}
    assert r.content[0].id != r.content[1].id

    assert from_ollama_response(chat("ok")).stop_reason == "end_turn"
    assert from_ollama_response(chat("cortado", done_reason="length")).stop_reason == "max_tokens"


# --- ponta a ponta com servidor falso -----------------------------------------


def test_agent_runs_a_full_task_against_the_local_model(fake_ollama, tmp_path):
    server = fake_ollama(
        [
            chat("Vou criar.", [call("write_file", path="oi.py", content="print('oi')\n")]),
            chat("Pronto."),
        ]
    )
    agent = Agent(
        Workspace(tmp_path),
        client=OllamaClient(host=server.url, num_ctx=8192),
        config=AgentConfig(model="qwen3:8b", auto_approve=True, max_tokens=512),
    )
    assert agent.run("crie oi.py") == "Pronto."
    assert (tmp_path / "oi.py").read_text() == "print('oi')\n"

    first, second = server.requests
    assert first["model"] == "qwen3:8b" and first["stream"] is False
    assert first["options"] == {"num_predict": 512, "num_ctx": 8192}
    assert {t["function"]["name"] for t in first["tools"]} >= {"read_file", "write_file"}
    assert first["messages"][0]["role"] == "system"
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_name"] == "write_file"
    assert "Created oi.py" in second["messages"][-1]["content"]


def test_unreachable_server_gives_a_friendly_error():
    client = OllamaClient(host="http://127.0.0.1:1")
    with pytest.raises(ProviderError, match="Ollama"):
        client.list_models()


def test_server_errors_are_reported(fake_ollama):
    server = fake_ollama([])  # sem respostas => 500
    with pytest.raises(ProviderError, match="500"):
        OllamaClient(host=server.url).post("/api/chat", {})


def test_host_without_scheme_and_has_model(fake_ollama):
    server = fake_ollama([], models=("qwen3:8b", "llama3.1:latest"))
    client = OllamaClient(host=server.url.removeprefix("http://"))
    assert client.has_model("qwen3:8b") and client.has_model("llama3.1")
    assert not client.has_model("mistral")


# --- CLI ----------------------------------------------------------------------


def test_provider_defaults_to_ollama_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert default_provider() == "ollama"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert default_provider() == "anthropic"


def test_cli_explicit_anthropic_requires_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--provider", "anthropic", "ola"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_without_key_uses_ollama_and_runs_a_task(monkeypatch, fake_ollama, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_CODE_AGENT_MODEL", raising=False)
    server = fake_ollama([chat("Olá do modelo local!")])
    code = main(["-w", str(tmp_path), "--host", server.url, "diga olá"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Olá do modelo local!" in out
    assert "(ollama)" in out


def test_cli_tells_how_to_download_a_missing_model(monkeypatch, fake_ollama, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server = fake_ollama([], models=("outro:1b",))
    assert main(["-w", str(tmp_path), "--host", server.url, "oi"]) == 2
    assert "ollama pull qwen3:8b" in capsys.readouterr().err


def test_cli_reports_ollama_not_running(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["-w", str(tmp_path), "--host", "http://127.0.0.1:1", "oi"]) == 2
    assert "Ollama" in capsys.readouterr().err
