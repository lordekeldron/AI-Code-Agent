import os

import pytest

from ai_code_agent.tools import (
    ToolError,
    build_tools,
    edit_file,
    execute_tool,
    is_blocked_command,
    list_dir,
    read_file,
    run_command,
    search,
    write_file,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="comandos de shell POSIX")


# --- confinamento ao workspace ------------------------------------------------


def test_path_escape_is_blocked(ws):
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file(ws, "../outside.txt")
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file(ws, "/etc/passwd")


@pytest.mark.skipif(os.name != "posix", reason="symlinks")
def test_symlink_escape_is_blocked(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret")
    (ws.root / "link.txt").symlink_to(outside)
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file(ws, "link.txt")


# --- read_file ----------------------------------------------------------------


def test_read_file_numbers_lines_and_supports_ranges(ws):
    (ws.root / "a.txt").write_text("one\ntwo\nthree\n")
    out = read_file(ws, "a.txt")
    assert "lines 1-3 of 3" in out
    assert "     1\tone" in out
    ranged = read_file(ws, "a.txt", start_line=2, end_line=2)
    assert "two" in ranged
    assert "one" not in ranged and "three" not in ranged


def test_read_file_errors_and_edge_cases(ws):
    with pytest.raises(ToolError, match="Not a file"):
        read_file(ws, "missing.txt")
    (ws.root / "empty.txt").write_text("")
    assert "empty" in read_file(ws, "empty.txt")
    (ws.root / "bin.dat").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ToolError, match="UTF-8"):
        read_file(ws, "bin.dat")
    (ws.root / "short.txt").write_text("x\n")
    with pytest.raises(ToolError, match="past the end"):
        read_file(ws, "short.txt", start_line=5)


def test_read_file_pages_large_files(ws):
    (ws.root / "big.txt").write_text("\n".join(f"l{i}" for i in range(1, 2501)))
    out = read_file(ws, "big.txt")
    assert "lines 1-2000 of 2500" in out
    assert "start_line=2001" in out


# --- write_file / edit_file -----------------------------------------------------


def test_write_file_creates_parent_dirs_and_overwrites(ws):
    assert "Created" in write_file(ws, "pkg/sub/mod.py", "x = 1\n")
    assert (ws.root / "pkg/sub/mod.py").read_text() == "x = 1\n"
    assert "Overwrote" in write_file(ws, "pkg/sub/mod.py", "x = 2\n")
    assert (ws.root / "pkg/sub/mod.py").read_text() == "x = 2\n"
    with pytest.raises(ToolError, match="directory"):
        write_file(ws, "pkg", "nope")


def test_edit_file_replaces_unique_text(ws):
    (ws.root / "f.py").write_text("a = 1\nb = 2\n")
    edit_file(ws, "f.py", "a = 1", "a = 10")
    assert (ws.root / "f.py").read_text() == "a = 10\nb = 2\n"


def test_edit_file_rejects_ambiguous_missing_and_noop(ws):
    (ws.root / "f.py").write_text("x\nx\n")
    with pytest.raises(ToolError, match="2 times"):
        edit_file(ws, "f.py", "x", "y")
    with pytest.raises(ToolError, match="not found"):
        edit_file(ws, "f.py", "zzz", "y")
    with pytest.raises(ToolError, match="identical"):
        edit_file(ws, "f.py", "x", "x")
    with pytest.raises(ToolError, match="must not be empty"):
        edit_file(ws, "f.py", "", "y")
    edit_file(ws, "f.py", "x", "y", replace_all=True)
    assert (ws.root / "f.py").read_text() == "y\ny\n"


def test_edit_file_preserves_crlf_line_endings(ws):
    (ws.root / "w.txt").write_bytes(b"a\r\nb\r\n")
    edit_file(ws, "w.txt", "a", "z")
    assert (ws.root / "w.txt").read_bytes() == b"z\r\nb\r\n"


# --- list_dir / search ----------------------------------------------------------


def test_list_dir_hides_noise_and_respects_depth(ws):
    (ws.root / "src").mkdir()
    (ws.root / "src" / "m.py").write_text("")
    (ws.root / ".git").mkdir()
    (ws.root / "node_modules").mkdir()
    (ws.root / "README.md").write_text("")
    shallow = list_dir(ws)
    assert "src/" in shallow and "README.md" in shallow
    assert ".git" not in shallow and "node_modules" not in shallow
    assert "m.py" not in shallow
    assert "m.py" in list_dir(ws, max_depth=2)


def test_list_dir_rejects_files(ws):
    (ws.root / "f.txt").write_text("")
    with pytest.raises(ToolError, match="Not a directory"):
        list_dir(ws, "f.txt")


def test_search_finds_matches_and_filters_by_glob(ws):
    (ws.root / "a.py").write_text("def hello():\n    pass\n")
    (ws.root / "b.txt").write_text("hello world\n")
    hits = search(ws, "hello")
    assert "a.py:1:" in hits and "b.txt:1:" in hits
    only_py = search(ws, "hello", glob="*.py")
    assert "a.py:1:" in only_py and "b.txt" not in only_py
    assert "a.py:1:" in search(ws, "HELLO", ignore_case=True)
    assert search(ws, "nothing-here") == "No matches."
    with pytest.raises(ToolError, match="Invalid regex"):
        search(ws, "(")
    with pytest.raises(ToolError, match="does not exist"):
        search(ws, "x", path="nope")


def test_search_skips_ignored_dirs_and_binary_files(ws):
    (ws.root / "node_modules").mkdir()
    (ws.root / "node_modules" / "lib.js").write_text("needle\n")
    (ws.root / "blob.bin").write_bytes(b"\xff\xfeneedle")
    assert search(ws, "needle") == "No matches."


# --- run_command ----------------------------------------------------------------


@posix_only
def test_run_command_captures_output_and_exit_code(ws):
    out = run_command(ws, "echo hi; echo err 1>&2; exit 3")
    assert "[exit code 3]" in out
    assert "hi" in out and "err" in out


@posix_only
def test_run_command_runs_in_workspace_root(ws):
    (ws.root / "marker.txt").write_text("")
    assert "marker.txt" in run_command(ws, "ls")


@posix_only
def test_run_command_times_out_and_kills_the_process(ws):
    out = run_command(ws, "sleep 5", timeout=1)
    assert "timed out" in out


@posix_only
def test_run_command_does_not_leak_the_api_key(ws, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    out = run_command(ws, "echo key=[$ANTHROPIC_API_KEY]")
    assert "sk-secret-value" not in out
    assert "key=[]" in out


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm -fr $HOME",
        "rm -r -f /",
        "sudo rm -rf /*",
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
        "dd if=/dev/zero of=/dev/sda",
    ],
)
def test_destructive_commands_are_blocked(ws, command):
    assert is_blocked_command(command)
    with pytest.raises(ToolError, match="blocked"):
        run_command(ws, command)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ./build",
        "rm -rf /tmp/some-dir",
        "rm -rf /home/user/project/dist",
        "rm notes.txt",
        "ls -la /",
        "python -m pytest",
        "git status",
    ],
)
def test_ordinary_commands_are_not_blocked(command):
    assert not is_blocked_command(command)


# --- registro e execução --------------------------------------------------------


def test_execute_tool_reports_errors_instead_of_raising(ws):
    tools = build_tools()
    text, is_error = execute_tool(ws, tools["read_file"], {"path": "nope.txt"})
    assert is_error and "Not a file" in text
    text, is_error = execute_tool(ws, tools["read_file"], {"wrong_arg": 1})
    assert is_error and "Invalid arguments" in text
    text, is_error = execute_tool(ws, tools["write_file"], {"path": "a.txt", "content": "hi"})
    assert not is_error and (ws.root / "a.txt").read_text() == "hi"


def test_tool_schemas_are_well_formed_and_approval_flags_are_right():
    tools = build_tools()
    for tool in tools.values():
        schema = tool.schema()
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["input_schema"]["type"] == "object"
        assert set(schema["input_schema"]["required"]) <= set(
            schema["input_schema"]["properties"]
        )
    assert {n for n, t in tools.items() if t.needs_approval} == {
        "write_file",
        "edit_file",
        "run_command",
    }
