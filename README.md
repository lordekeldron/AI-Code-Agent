# AI Code Agent

Agente de IA para programação. Você descreve uma tarefa em linguagem natural e ele explora o projeto, lê e edita arquivos, roda comandos (testes, linters, git) e repete até concluir — tudo dentro de um diretório de trabalho (o *workspace*).

Funciona com o loop de *tool use* da API da Anthropic: o modelo decide qual ferramenta chamar, o agente executa e devolve o resultado, até o modelo terminar.

## Instalação

Requer Python 3.10+ e uma chave da API da Anthropic.

```bash
git clone https://github.com/lordekeldron/AI-Code-Agent
cd AI-Code-Agent
pip install -e .            # ou: pip install -e ".[dev]" para rodar os testes
export ANTHROPIC_API_KEY="sua-chave"
```

## Uso

```bash
# modo interativo, no diretório atual
ai-code-agent

# uma tarefa só, em outro projeto
ai-code-agent -w ~/meu-projeto "adicione testes para a função parse_config e rode a suíte"

# também funciona como módulo
python -m ai_code_agent --help
```

Opções principais:

| Opção | Descrição |
|---|---|
| `-w, --workspace DIR` | Diretório do projeto (padrão: o atual) |
| `-m, --model NOME` | Modelo a usar (padrão `claude-sonnet-5-5`; ou a variável `AI_CODE_AGENT_MODEL`) |
| `-y, --yes` | Aprova edições e comandos automaticamente (use com cuidado) |
| `--max-turns N` | Máximo de passos por tarefa (padrão 40) |
| `--max-tokens N` | Máximo de tokens por resposta (padrão 8192) |

No modo interativo: `/reset` esquece a conversa, `/ajuda` lista os comandos e `/sair` encerra.

## Ferramentas do agente

| Ferramenta | O que faz | Pede aprovação |
|---|---|---|
| `list_dir` | Lista arquivos e pastas (esconde `.git`, `node_modules` etc.) | não |
| `read_file` | Lê um arquivo texto com números de linha, em blocos | não |
| `search` | Busca por regex no conteúdo dos arquivos | não |
| `write_file` | Cria ou sobrescreve um arquivo | **sim** |
| `edit_file` | Troca um trecho exato de um arquivo existente | **sim** |
| `run_command` | Executa um comando de shell no workspace (com timeout) | **sim** |

## Segurança

- **Confinamento de arquivos:** as ferramentas de arquivo recusam qualquer caminho fora do workspace, incluindo `..` e *symlinks* que apontem para fora.
- **Aprovação:** escrever, editar e executar comandos pedem confirmação no terminal (`s`/`n`/`t` para todas na sessão). Sem um callback de aprovação, a biblioteca nega essas ações por padrão.
- **Chave da API:** é removida do ambiente dos comandos que o agente executa.
- **Comandos bloqueados:** há uma lista de melhor esforço que barra operações destrutivas óbvias (`rm -rf /`, `mkfs`, fork bomb etc.).

> ⚠️ `run_command` executa um shell de verdade na sua máquina e **não é um sandbox**. A proteção real é você ler o comando antes de aprovar. Para rodar com `--yes` ou em projetos que você não conhece, use um container ou uma VM.

## Usando como biblioteca

```python
from ai_code_agent import Agent, AgentConfig, Hooks, Workspace

agent = Agent(
    Workspace("~/meu-projeto"),
    config=AgentConfig(max_turns=30),
    hooks=Hooks(
        on_text=print,
        approve=lambda name, args: input(f"Permitir {name}? [s/N] ") == "s",
    ),
)
agent.run("Corrija o bug no módulo de login e rode os testes")
```

`client=` aceita qualquer objeto com `messages.create(...)`, o que facilita testes com um cliente simulado.

## Estrutura

```
src/ai_code_agent/
├── agent.py   # loop agêntico, prompt de sistema, histórico e rollback em falhas
├── tools.py   # workspace confinado + as 6 ferramentas + registro/execução
└── cli.py     # REPL, confirmações no terminal e opções de linha de comando
tests/         # testes das ferramentas e do loop (cliente simulado)
```

## Testes

```bash
pip install -e ".[dev]"
pytest
```

Os testes não chamam a API: o loop é validado com um cliente simulado.

## Limitações e próximos passos

- Sem *streaming* das respostas (o texto aparece quando cada passo termina).
- Sem compactação de contexto: conversas muito longas acabam enchendo a janela do modelo.
- Ferramentas de arquivo só lidam com texto UTF-8.
- Ideias: streaming, resumo automático do histórico, ferramenta de diff/`git` dedicada, execução de comandos em container.
