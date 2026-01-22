"""
Configuration for CLI Coding Agent.

Supports LiteLLM proxy for unified LLM access across providers.
Implements configuration for all 7 architectural layers.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers through LiteLLM."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LITELLM_PROXY = "litellm_proxy"


class AgentRole(str, Enum):
    """Specialist agent roles based on MetaGPT/OpenHands patterns."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    ARCHITECT = "architect"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    DEBUGGER = "debugger"


class MemoryType(str, Enum):
    """Memory system types from research."""
    SIMPLE_MEM = "simple_mem"  # arxiv:2601.02553
    A_MEM = "a_mem"  # Zettelkasten-style, NeurIPS 2025
    AGE_MEM = "age_mem"  # RL-based, arxiv:2601.01885
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class PlanningStrategy(str, Enum):
    """Planning strategies from research."""
    MCTS = "mcts"  # Monte Carlo Tree Search / LATS
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    LOONG_FLOW_PES = "loong_flow_pes"  # Plan/Execute/Summary
    BEST_OF_N = "best_of_n"
    MULTI_ISLAND = "multi_island"  # Evolutionary with MAP-Elites


class LLMConfig(BaseModel):
    """LLM configuration for LiteLLM proxy integration."""
    provider: LLMProvider = LLMProvider.LITELLM_PROXY
    model: str = Field(default="gpt-4o", description="Model name or LiteLLM model string")
    api_base: Optional[str] = Field(default=None, description="LiteLLM proxy URL or provider base URL")
    api_key: Optional[str] = Field(default=None, description="API key (uses env var if not set)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3)

    # Cost tracking
    track_costs: bool = True
    cost_limit_per_task: Optional[float] = Field(default=None, description="Max cost per task in USD")

    # Context management
    context_window: int = Field(default=128000, description="Model context window size")
    summarization_threshold: float = Field(default=0.7, description="Trigger summarization at this % of context")

    class Config:
        use_enum_values = True


class RLMConfig(BaseModel):
    """
    Recursive Language Model configuration (arxiv:2512.24601).

    Context stored as external REPL variable, LLM writes code to manipulate it.
    """
    enabled: bool = True
    repl_timeout: int = Field(default=30, description="REPL execution timeout in seconds")
    max_recursion_depth: int = Field(default=5, description="Max recursive sub-model calls")
    sub_model: Optional[str] = Field(default=None, description="Smaller model for recursive calls")
    chunk_size: int = Field(default=4000, description="Default chunk size for context decomposition")
    aggregation_strategy: str = Field(default="hierarchical", description="How to aggregate recursive results")


class SimpleMemConfig(BaseModel):
    """
    SimpleMem configuration (arxiv:2601.02553).

    Three-stage pipeline: Compression, Consolidation, Retrieval.
    """
    enabled: bool = True

    # Stage 1: Semantic Structured Compression
    compression_model: Optional[str] = Field(default=None, description="Model for compression, uses main if None")
    entropy_threshold: float = Field(default=0.3, description="Filter low-entropy content")
    coreference_resolution: bool = True  # Replace pronouns with actual names
    temporal_anchoring: bool = True  # Convert relative to absolute timestamps

    # Stage 2: Recursive Memory Consolidation
    consolidation_interval: int = Field(default=10, description="Consolidate every N memory units")
    similarity_threshold: float = Field(default=0.85, description="Merge memories above this similarity")
    max_abstraction_levels: int = Field(default=3)

    # Stage 3: Adaptive Query-Aware Retrieval
    min_retrieval_scope: int = Field(default=3, description="Min memories to retrieve")
    max_retrieval_scope: int = Field(default=20, description="Max memories to retrieve")
    query_complexity_model: bool = True  # Use LLM to assess query complexity


class AMemConfig(BaseModel):
    """
    A-MEM configuration (NeurIPS 2025).

    Zettelkasten-inspired interconnected knowledge networks.
    Uses ~2,000 tokens vs MemGPT's ~16,900.
    """
    enabled: bool = True
    max_memory_units: int = Field(default=1000)
    link_threshold: float = Field(default=0.7, description="Semantic similarity for auto-linking")
    enable_dynamic_linking: bool = True  # New memories trigger updates to existing
    embedding_model: str = Field(default="text-embedding-3-small")


class AgeMemConfig(BaseModel):
    """
    AgeMem configuration (arxiv:2601.01885).

    RL-based LTM/STM management with tool-based operations.
    """
    enabled: bool = False  # Optional advanced feature
    operations: list[str] = Field(
        default=["STORE", "RETRIEVE", "UPDATE", "SUMMARIZE", "DELETE", "FILTER"]
    )
    stm_capacity: int = Field(default=10, description="Short-term memory capacity")
    ltm_capacity: int = Field(default=1000, description="Long-term memory capacity")
    use_rl_policy: bool = False  # Requires training


class MemoryConfig(BaseModel):
    """Combined memory configuration."""
    simple_mem: SimpleMemConfig = Field(default_factory=SimpleMemConfig)
    a_mem: AMemConfig = Field(default_factory=AMemConfig)
    age_mem: AgeMemConfig = Field(default_factory=AgeMemConfig)

    # Storage backends
    vector_store: str = Field(default="chroma", description="chroma, qdrant, faiss")
    vector_store_path: Optional[str] = Field(default=None, description="Path for persistent storage")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Hierarchical memory separation
    separate_orchestrator_context: bool = True  # Subagents don't inherit full history
    artifacts_as_handles: bool = True  # Large data stored externally with references


class CodeUnderstandingConfig(BaseModel):
    """
    Code understanding configuration.

    Combines Tree-sitter, LSP, SCIP, and Graphiti.
    """
    # Tree-sitter AST parsing
    tree_sitter_enabled: bool = True
    supported_languages: list[str] = Field(
        default=["python", "javascript", "typescript", "go", "rust", "java", "c", "cpp"]
    )

    # LSP integration (50ms vs 45s text search - 900x improvement)
    lsp_enabled: bool = True
    lsp_servers: dict[str, str] = Field(
        default={
            "python": "pylsp",
            "typescript": "typescript-language-server",
            "javascript": "typescript-language-server",
            "go": "gopls",
            "rust": "rust-analyzer",
        }
    )

    # SCIP protocol (10x faster than LSIF)
    scip_enabled: bool = True
    scip_index_path: Optional[str] = Field(default=None)
    incremental_indexing: bool = True

    # Graphiti knowledge graph (90% latency reduction vs baseline RAG)
    graphiti_enabled: bool = True
    graph_store: str = Field(default="networkx", description="networkx, neo4j, memgraph")
    temporal_awareness: bool = True  # Track when facts were learned

    # LSPRAG (LSP-Guided RAG)
    lsprag_enabled: bool = True

    # Code embeddings
    code_embedding_model: str = Field(default="microsoft/unixcoder-base")  # Open-source alternative to voyage-code-3


class PlanningConfig(BaseModel):
    """
    Planning configuration implementing multiple strategies.
    """
    default_strategy: PlanningStrategy = PlanningStrategy.LOONG_FLOW_PES

    # MCTS / LATS configuration
    mcts_simulations: int = Field(default=10, description="Number of MCTS simulations")
    mcts_exploration_weight: float = Field(default=1.414, description="UCB exploration constant")
    mcts_max_depth: int = Field(default=10)

    # Tree of Thoughts
    tot_num_thoughts: int = Field(default=3, description="Thoughts to generate per step")
    tot_evaluation_strategy: str = Field(default="vote", description="vote, score, debate")

    # LoongFlow PES (Plan/Execute/Summary)
    pes_require_plan_approval: bool = False  # Require human approval of plans
    pes_verification_contracts: bool = True  # Code must pass contracts
    pes_abductive_reasoning: bool = True  # Store lessons from failures

    # Best-of-N with verification
    best_of_n_candidates: int = Field(default=3)
    best_of_n_verifier: str = Field(default="tests", description="tests, linter, llm")

    # Multi-Island evolutionary strategy
    multi_island_agents: int = Field(default=3, description="Parallel island agents")
    map_elites_archive: bool = True  # Store partial solutions from failures
    crossover_enabled: bool = True  # Synthesize from multiple islands

    # VeriPlan formal verification
    veriplan_enabled: bool = False  # Requires formal spec support
    veriplan_checker: Optional[str] = Field(default=None, description="Model checker to use")


class ValidationConfig(BaseModel):
    """
    Validation pipeline configuration.

    Implements AgentCoder and Multi-Agent Reflexion patterns.
    """
    # AgentCoder pattern (81.8% success rate)
    agent_coder_enabled: bool = True
    programmer_agent_model: Optional[str] = None  # Uses main model if None
    test_designer_agent_model: Optional[str] = None
    test_executor_deterministic: bool = True  # Always use deterministic executor

    # Multi-Agent Reflexion (MAR) - arxiv:2512.20845
    mar_enabled: bool = True
    mar_personas: list[str] = Field(
        default=["actor", "diagnostician", "critic", "aggregator"]
    )
    mar_max_iterations: int = Field(default=5)

    # Reflexion self-improvement
    reflexion_enabled: bool = True
    reflexion_memory_window: int = Field(default=3, description="Past attempts to remember")

    # Deterministic tools for verification
    use_deterministic_file_check: bool = True  # os.path.exists() not LLM
    use_deterministic_git_status: bool = True  # git status --porcelain
    use_deterministic_test_runner: bool = True  # pytest --json
    use_deterministic_linter: bool = True  # ruff, eslint, etc.

    # Confidence thresholds
    confidence_threshold: float = Field(default=0.8, description="Escalate below this")
    require_tests_for_changes: bool = True  # No code changes without tests


class SandboxConfig(BaseModel):
    """
    Execution environment configuration.

    Uses DeepAgents Sandbox (not custom Docker).
    """
    # DeepAgents built-in sandbox
    use_deepagents_sandbox: bool = True

    # Fallback Docker config
    docker_enabled: bool = False
    docker_image: str = Field(default="python:3.11-slim")
    docker_network_disabled: bool = True  # Security: disable network
    docker_memory_limit: str = Field(default="2g")
    docker_cpu_limit: float = Field(default=2.0)
    docker_timeout: int = Field(default=300, description="Execution timeout in seconds")

    # Transactional filesystem snapshots (100% interception, ~14.5% overhead)
    transactional_fs: bool = True
    snapshot_before_changes: bool = True
    auto_rollback_on_failure: bool = True

    # Git worktrees for parallel work
    git_worktrees_enabled: bool = True
    worktree_base_path: Optional[str] = Field(default=None)

    # Security
    run_as_non_root: bool = True
    allowed_commands: list[str] = Field(default=[])  # Empty = all allowed
    blocked_commands: list[str] = Field(
        default=["rm -rf /", "sudo", "chmod 777", "curl | sh", "wget | sh"]
    )


class ObservabilityConfig(BaseModel):
    """
    Observability configuration.

    OpenTelemetry + Langfuse + Loop Detection.
    """
    # OpenTelemetry
    otel_enabled: bool = True
    otel_endpoint: Optional[str] = Field(default=None)
    otel_service_name: str = Field(default="cli-coding-agent")

    # Langfuse tracing
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Metrics to track
    track_token_consumption: bool = True
    track_tool_call_latency: bool = True
    track_reasoning_steps: bool = True
    track_cost_per_task: bool = True

    # Loop detection for runaway agents
    loop_detection_enabled: bool = True
    loop_detection_window: int = Field(default=10, description="Actions to check for loops")
    loop_similarity_threshold: float = Field(default=0.9)
    max_consecutive_failures: int = Field(default=5)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    log_format: str = Field(default="json")


class UIConfig(BaseModel):
    """UI/UX configuration for CLI."""
    # Playwright MCP for UI testing
    playwright_enabled: bool = True
    playwright_use_accessibility_tree: bool = True  # Not screenshots
    playwright_self_healing: bool = True  # Auto-update selectors
    playwright_headless: bool = True

    # Rich terminal UI
    rich_enabled: bool = True
    show_progress: bool = True
    show_token_usage: bool = True
    show_cost: bool = True
    theme: str = Field(default="monokai")

    # Interactive mode
    interactive: bool = True
    confirm_destructive: bool = True  # Confirm rm, git push, etc.
    auto_approve_safe: bool = True  # Auto-approve read-only operations


class AgentConfig(BaseModel):
    """
    Master configuration for CLI Coding Agent.

    Combines all 7 architectural layers from research synthesis.
    """
    # Basic settings
    project_root: Optional[str] = Field(default=None, description="Project to work on")
    workspace_dir: str = Field(default=".cli-agent", description="Agent workspace directory")

    # Layer configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rlm: RLMConfig = Field(default_factory=RLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    code_understanding: CodeUnderstandingConfig = Field(default_factory=CodeUnderstandingConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    # Agent roles to enable
    enabled_roles: list[AgentRole] = Field(
        default=[
            AgentRole.ORCHESTRATOR,
            AgentRole.PLANNER,
            AgentRole.CODER,
            AgentRole.TESTER,
            AgentRole.REVIEWER,
        ]
    )

    # Live-SWE-Agent style self-evolution
    self_evolution_enabled: bool = False  # Agent can modify its own capabilities

    class Config:
        use_enum_values = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        config = cls()

        # LLM settings from env
        if api_key := os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = api_key
        if api_base := os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE"):
            config.llm.api_base = api_base
        if model := os.getenv("CLI_AGENT_MODEL"):
            config.llm.model = model

        # Langfuse from env
        if langfuse_pk := os.getenv("LANGFUSE_PUBLIC_KEY"):
            config.observability.langfuse_enabled = True
            config.observability.langfuse_public_key = langfuse_pk
        if langfuse_sk := os.getenv("LANGFUSE_SECRET_KEY"):
            config.observability.langfuse_secret_key = langfuse_sk

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_litellm_kwargs(self) -> dict[str, Any]:
        """Get kwargs for LiteLLM completion calls."""
        kwargs = {
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "top_p": self.llm.top_p,
            "timeout": self.llm.timeout,
            "num_retries": self.llm.max_retries,
        }

        if self.llm.api_base:
            kwargs["api_base"] = self.llm.api_base
        if self.llm.api_key:
            kwargs["api_key"] = self.llm.api_key

        return kwargs


def load_config(path: Optional[str | Path] = None) -> AgentConfig:
    """
    Load configuration from file or environment.

    Args:
        path: Optional path to YAML config file. If None, loads from environment.

    Returns:
        AgentConfig instance
    """
    if path:
        return AgentConfig.from_yaml(path)
    return AgentConfig.from_env()
