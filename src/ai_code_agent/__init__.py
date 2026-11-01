"""AI Code Agent: um agente de código baseado no loop de tool use da API da Anthropic."""

from .agent import Agent, AgentConfig, Hooks
from .tools import Workspace, build_tools

__all__ = ["Agent", "AgentConfig", "Hooks", "Workspace", "build_tools"]
__version__ = "0.1.0"
