import copy
from types import SimpleNamespace

import pytest

from ai_code_agent import Agent, AgentConfig, Hooks, Workspace
from ai_code_agent.cli import describe_call


class FakeMessages:
    """Imita client.messages.create devolvendo respostas pré-definidas."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))  # o agente muta a lista depois
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(tool_id, name, **arguments):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=arguments)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


@pytest.fixture
def make_agent(tmp_path):
    def _make(responses, *, hooks=None, **config):
        client = FakeClient(responses)
        agent = Agent(
            Workspace(tmp_path), client=client, config=AgentConfig(**config), hooks=hooks
        )
        return agent, client

    return _make


def test_plain_answer_without_tools(make_agent):
    agent, client = make_agent([response("end_turn", text("Olá!"))])
    assert agent.run("oi") == "Olá!"
    assert agent.messages == [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Olá!"}]},
    ]
    request = client.messages.calls[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert {t["name"] for t in request["tools"]} >= {"read_file", "edit_file", "run_command"}
    assert str(agent.workspace.root) in request["system"][0]["text"]


def test_tool_loop_runs_tools_and_feeds_results_back(make_agent, tmp_path):
    agent, client = make_agent(
        [
            response(
                "tool_use",
                text("Vou criar o arquivo."),
                tool_use("t1", "write_file", path="hello.py", content="print('oi')\n"),
            ),
            response("end_turn", text("Pronto.")),
        ],
        auto_approve=True,
    )
    assert agent.run("crie hello.py") == "Pronto."
    assert (tmp_path / "hello.py").read_text() == "print('oi')\n"

    second_request = client.messages.calls[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    assert second_request[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "write_file",
        "input": {"path": "hello.py", "content": "print('oi')\n"},
    }
    tool_result = second_request[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "t1"
    assert "Created hello.py" in tool_result["content"]
    assert "is_error" not in tool_result


def test_denied_action_is_not_executed(make_agent, tmp_path):
    agent, client = make_agent(
        [
            response("tool_use", tool_use("t1", "write_file", path="x.txt", content="x")),
            response("end_turn", text("Entendi, não vou escrever.")),
        ]
    )  # sem hook approve => nega tudo
    agent.run("escreva x.txt")
    assert not (tmp_path / "x.txt").exists()
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "denied" in result["content"]


def test_approve_hook_can_allow_an_action(make_agent, tmp_path):
    asked = []

    def approve(name, args):
        asked.append((name, args["path"]))
        return True

    agent, _ = make_agent(
        [
            response("tool_use", tool_use("t1", "write_file", path="x.txt", content="x")),
            response("end_turn", text("ok")),
        ],
        hooks=Hooks(approve=approve),
    )
    agent.run("escreva x.txt")
    assert asked == [("write_file", "x.txt")]
    assert (tmp_path / "x.txt").read_text() == "x"


def test_read_only_tools_never_ask_for_approval(make_agent, tmp_path):
    (tmp_path / "notes.txt").write_text("conteudo secreto\n")
    agent, client = make_agent(
        [
            response("tool_use", tool_use("t1", "read_file", path="notes.txt")),
            response("end_turn", text("lido")),
        ]
    )
    agent.run("leia notes.txt")
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert "conteudo secreto" in result["content"]
    assert "is_error" not in result


def test_unknown_tool_and_tool_errors_go_back_to_the_model(make_agent):
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use("t1", "teleport", where="mars"),
                tool_use("t2", "read_file", path="nope.txt"),
            ),
            response("end_turn", text("não deu")),
        ]
    )
    agent.run("faça algo")
    results = client.messages.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]  # um resultado por tool_use
    assert all(r["is_error"] for r in results)
    assert "Unknown tool" in results[0]["content"]
    assert "Not a file" in results[1]["content"]


def test_hooks_receive_text_tool_calls_and_results(make_agent, tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    events = []
    hooks = Hooks(
        on_text=lambda t: events.append(("text", t)),
        on_tool_use=lambda n, a: events.append(("use", n)),
        on_tool_result=lambda n, t, e: events.append(("result", n, e)),
    )
    agent, _ = make_agent(
        [
            response("tool_use", text("lendo"), tool_use("t1", "read_file", path="a.txt")),
            response("end_turn", text("fim")),
        ],
        hooks=hooks,
    )
    agent.run("leia")
    assert events == [
        ("text", "lendo"),
        ("use", "read_file"),
        ("result", "read_file", False),
        ("text", "fim"),
    ]


def test_max_turns_stops_the_loop_with_a_notice(make_agent, tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    notices = []
    looping = [
        response("tool_use", tool_use(f"t{i}", "read_file", path="a.txt")) for i in range(5)
    ]
    agent, client = make_agent(
        looping, hooks=Hooks(on_notice=notices.append), max_turns=2
    )
    agent.run("em loop")
    assert len(client.messages.calls) == 2
    assert len(notices) == 1 and "2 passos" in notices[0]


def test_history_is_rolled_back_when_a_call_fails(make_agent, tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    agent, _ = make_agent(
        [
            response("tool_use", tool_use("t1", "read_file", path="a.txt")),
            RuntimeError("boom"),
            response("end_turn", text("ok")),
        ]
    )
    with pytest.raises(RuntimeError, match="boom"):
        agent.run("primeira")
    assert agent.messages == []  # nada de tool_use sem tool_result pendurado
    assert agent.run("segunda") == "ok"
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_unexecuted_tool_use_is_dropped_when_output_is_cut_off(make_agent):
    notices = []
    agent, _ = make_agent(
        [
            response(
                "max_tokens",
                text("parcial"),
                tool_use("t1", "write_file", path="x.txt", content="inco"),
            )
        ],
        hooks=Hooks(on_notice=notices.append),
    )
    assert agent.run("escreva") == "parcial"
    assert agent.messages[-1]["content"] == [{"type": "text", "text": "parcial"}]
    assert notices and "max_tokens" in notices[0]


def test_blank_text_blocks_are_never_stored(make_agent):
    agent, _ = make_agent([response("end_turn", text("   "))])
    assert agent.run("oi") == ""
    assert agent.messages[-1]["content"] == [{"type": "text", "text": "(no output)"}]


def test_reset_clears_the_conversation(make_agent):
    agent, _ = make_agent([response("end_turn", text("oi"))])
    agent.run("a")
    agent.reset()
    assert agent.messages == []


# --- CLI ----------------------------------------------------------------------


def test_describe_call_shows_what_will_happen():
    assert "$ pytest -q" in describe_call("run_command", {"command": "pytest -q"})
    assert "x.py" in describe_call("write_file", {"path": "x.py", "content": "a = 1"})
    edit = describe_call("edit_file", {"path": "x.py", "old_string": "a", "new_string": "b"})
    assert "--- remover" in edit and "+++ inserir" in edit
