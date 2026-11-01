import pytest

from ai_code_agent.tools import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)
