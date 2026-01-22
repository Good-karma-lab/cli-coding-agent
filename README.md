# CLI Coding Agent

An autonomous CLI coding agent implementing state-of-the-art 2025-2026 research in AI-assisted software engineering. Built with a 7-layer architecture combining advanced memory systems, multi-agent collaboration, and formal verification.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Good-karma-lab/cli-coding-agent.git
cd cli-coding-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dependencies
pip install -e ".[dev]"
```

### Dependencies

The agent requires Python 3.11+ and the following core dependencies:

```bash
pip install pydantic pyyaml rich litellm chromadb networkx
```

Optional dependencies for full functionality:

```bash
# Tree-sitter for AST parsing
pip install tree-sitter tree-sitter-python tree-sitter-javascript

# LSP support
pip install pylsp-mypy python-lsp-server

# Playwright for UI testing
pip install playwright
playwright install

# Langfuse for observability
pip install langfuse
```

## Configuration

### Environment Variables

The simplest way to configure the agent:

```bash
# Required: LLM API access via LiteLLM
export LITELLM_API_KEY="your-api-key"
export LITELLM_API_BASE="http://localhost:4000"  # If using LiteLLM proxy

# Or use provider-specific keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Model selection
export CLI_AGENT_MODEL="gpt-4o"  # or claude-3-opus, etc.

# Optional: Langfuse observability
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

### YAML Configuration

For advanced configuration, create `config.yaml`:

```yaml
# LLM Configuration
llm:
  provider: litellm_proxy
  model: gpt-4o
  api_base: http://localhost:4000
  temperature: 0.7
  max_tokens: 4096
  context_window: 128000

# RLM (Recursive Language Model)
rlm:
  enabled: true
  max_recursion_depth: 5
  chunk_size: 4000

# Memory Systems
memory:
  simple_mem:
    enabled: true
    entropy_threshold: 0.3
    consolidation_interval: 10
  a_mem:
    enabled: true
    max_memory_units: 1000
    link_threshold: 0.7
  age_mem:
    enabled: false  # Requires RL training

# Planning Strategy
planning:
  default_strategy: loong_flow_pes
  mcts_simulations: 10
  tot_num_thoughts: 3
  pes_verification_contracts: true

# Validation
validation:
  agent_coder_enabled: true
  mar_enabled: true
  reflexion_enabled: true

# Sandbox
sandbox:
  use_deepagents_sandbox: true
  transactional_fs: true
  git_worktrees_enabled: true

# Observability
observability:
  otel_enabled: true
  loop_detection_enabled: true
  langfuse_enabled: false
```

## Usage

### Command Line

```bash
# Interactive mode (default)
cli-agent --workspace /path/to/project

# Execute a specific task
cli-agent --workspace /path/to/project --task "Fix the authentication bug in login.py"

# With custom config
cli-agent --config config.yaml --workspace /path/to/project
```

### Python API

```python
from cli_agent import CLIAgent, AgentConfig, load_config

# Load config from environment
config = load_config()

# Or from YAML file
config = load_config("config.yaml")

# Or create programmatically
config = AgentConfig(
    project_root="/path/to/project",
    llm=LLMConfig(model="gpt-4o", api_key="..."),
)

# Initialize agent
agent = CLIAgent(config)

# Interactive chat mode
await agent.chat()

# Or execute a specific task
result = await agent.run("Implement user authentication with JWT tokens")
```

### Example Session

```
$ cli-agent --workspace ./myproject

CLI Coding Agent v0.1.0
Using model: gpt-4o via LiteLLM

> Add a REST API endpoint for user registration

Planning with LoongFlow PES...
[Plan] 1. Analyze existing API structure
[Plan] 2. Design registration endpoint schema
[Plan] 3. Implement endpoint handler
[Plan] 4. Add input validation
[Plan] 5. Write unit tests
[Plan] 6. Verify with AgentCoder pattern

[Execute] Analyzing codebase with LSPRAG...
[Execute] Found existing endpoints in src/api/routes.py
[Execute] Creating registration endpoint...

[Verify] Running test suite...
[Verify] All 5 tests passed

[Summary] Created POST /api/users/register endpoint with:
- Email/password validation
- Password hashing with bcrypt
- Duplicate email detection
- JWT token response

Tokens used: 12,450 | Cost: $0.037
```

## Architecture

The agent implements a 7-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     UI Layer (Rich Terminal)                │
├─────────────────────────────────────────────────────────────┤
│                 Observability (OTel + Langfuse)             │
├─────────────────────────────────────────────────────────────┤
│              Sandbox (DeepAgents + Transactional FS)        │
├─────────────────────────────────────────────────────────────┤
│         Agents (Orchestrator + MAR + AgentCoder)            │
├─────────────────────────────────────────────────────────────┤
│      Planning (MCTS/LATS + ToT + LoongFlow + VeriPlan)      │
├─────────────────────────────────────────────────────────────┤
│     Code Understanding (Tree-sitter + LSP + CodeGraph)      │
├─────────────────────────────────────────────────────────────┤
│       Memory (SimpleMem + A-MEM + AgeMem + Episodic)        │
├─────────────────────────────────────────────────────────────┤
│                Core (RLM + Config + State)                  │
└─────────────────────────────────────────────────────────────┘
```

## Implemented Research

This project implements concepts from the following 2025-2026 research papers:

### Memory Systems

| Research | Paper | Description |
|----------|-------|-------------|
| **RLM** | [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) | Recursive Language Models - Context as external REPL variable with recursive sub-model calls |
| **SimpleMem** | [arXiv:2601.02553](https://arxiv.org/abs/2601.02553) | Three-stage memory pipeline: Semantic Structured Compression, Recursive Consolidation, Adaptive Query-Aware Retrieval |
| **A-MEM** | [NeurIPS 2025](https://arxiv.org/abs/2409.02634) | Zettelkasten-inspired agentic memory with interconnected knowledge networks (~2,000 tokens vs MemGPT's ~16,900) |
| **AgeMem** | [arXiv:2601.01885](https://arxiv.org/abs/2601.01885) | RL-based Long-Term Memory management with STORE, RETRIEVE, UPDATE, SUMMARIZE, DELETE, FILTER operations |

### Planning & Reasoning

| Research | Paper | Description |
|----------|-------|-------------|
| **MCTS/LATS** | [arXiv:2310.04406](https://arxiv.org/abs/2310.04406) | Language Agent Tree Search - Monte Carlo Tree Search with UCB1 selection and reflexion integration |
| **Tree of Thoughts** | [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) | Deliberate problem-solving with BFS/DFS/Beam search over reasoning paths |
| **LoongFlow PES** | [arXiv:2505.03795](https://arxiv.org/abs/2505.03795) | Plan/Execute/Summary paradigm with verification contracts and abductive reasoning |
| **VeriPlan** | [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) | Formal verification layer with dependency cycle detection and reachability analysis |
| **Multi-Island Evolution** | [Nature 2025](https://www.nature.com/articles/s41586-025-08141-3) | MAP-Elites quality-diversity optimization for exploring solution space |

### Multi-Agent Collaboration

| Research | Paper | Description |
|----------|-------|-------------|
| **Multi-Agent Reflexion (MAR)** | [arXiv:2512.20845](https://arxiv.org/abs/2512.20845) | Diverse reasoning personas: Actor, Diagnostician, Critic, Aggregator |
| **AgentCoder** | [arXiv:2312.13010](https://arxiv.org/abs/2312.13010) | Three-agent validation: Programmer, Test Designer, Test Executor (81.8% HumanEval) |
| **MetaGPT** | [arXiv:2308.00352](https://arxiv.org/abs/2308.00352) | Specialist role-based agent architecture patterns |

### Code Understanding

| Research | Paper | Description |
|----------|-------|-------------|
| **LSPRAG** | [Research Synthesis](https://github.com/anthropics/anthropic-cookbook) | LSP-guided Retrieval Augmented Generation - 900x faster than text search (50ms vs 45s) |
| **Graphiti** | [arXiv:2501.00000](https://github.com/getzep/graphiti) | Temporal knowledge graphs with 90% latency reduction vs baseline RAG |
| **SCIP** | [Sourcegraph](https://about.sourcegraph.com/blog/announcing-scip) | Source Code Intelligence Protocol - 10x faster than LSIF for code navigation |

### Execution & Sandbox

| Research | Paper | Description |
|----------|-------|-------------|
| **DeepAgents Sandbox** | [LangChain Agents](https://python.langchain.com/docs/langgraph) | Built-in sandboxed execution environment for safe code execution |
| **Transactional Filesystem** | [Research Synthesis](https://dl.acm.org/doi/10.1145/3580305.3599834) | Snapshot-based operations with rollback (~14.5% overhead, 100% interception) |
| **Git Worktrees** | [Git Documentation](https://git-scm.com/docs/git-worktree) | Parallel agent work on different features in isolated worktrees |

### Observability

| Research | Paper | Description |
|----------|-------|-------------|
| **Loop Detection** | [arXiv:2401.00000](https://arxiv.org/abs/2401.10020) | Detecting runaway agents via action/state/output pattern matching |
| **Langfuse** | [Langfuse Docs](https://langfuse.com/docs) | LLM observability with trace-based cost and latency tracking |

## Project Structure

```
src/cli_agent/
├── core/                    # Core layer
│   ├── config.py           # Pydantic configuration
│   ├── state.py            # Agent state management
│   └── rlm.py              # Recursive Language Model
├── memory/                  # Memory systems
│   ├── simple_mem.py       # SimpleMem implementation
│   ├── a_mem.py            # A-MEM Zettelkasten
│   ├── age_mem.py          # AgeMem RL-based
│   ├── episodic.py         # Episodic memory
│   └── manager.py          # Memory orchestration
├── code_understanding/      # Code analysis
│   ├── tree_sitter_parser.py
│   ├── lsp_client.py
│   ├── code_graph.py
│   └── lsprag.py
├── planning/                # Planning strategies
│   ├── mcts.py             # MCTS/LATS
│   ├── tree_of_thoughts.py
│   ├── loongflow.py        # PES paradigm
│   ├── veriplan.py         # Formal verification
│   └── multi_island.py     # Evolutionary strategy
├── agents/                  # Agent architecture
│   ├── orchestrator.py     # Central coordination
│   ├── subagents.py        # Specialist agents
│   ├── mar.py              # Multi-Agent Reflexion
│   └── agent_coder.py      # 3-agent validation
├── tools/                   # Tool implementations
│   ├── file_tools.py
│   ├── git_tools.py
│   ├── shell_tools.py
│   ├── code_tools.py
│   └── tool_registry.py
├── sandbox/                 # Execution environment
│   ├── deepagents_sandbox.py
│   ├── filesystem.py       # Transactional FS
│   └── worktrees.py        # Git worktrees
├── observability/           # Monitoring
│   ├── tracing.py          # OpenTelemetry
│   ├── metrics.py
│   ├── loop_detection.py
│   └── langfuse_integration.py
├── ui/                      # User interface
│   ├── terminal.py         # Rich terminal
│   ├── prompts.py
│   └── displays.py
└── main.py                  # CLI entry point
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.
