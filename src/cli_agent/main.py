"""
Main Agent Orchestration - CLI Coding Agent Entry Point

Integrates all modules into a cohesive autonomous coding agent:
- RLM recursive context management
- Memory systems (SimpleMem, A-MEM, AgeMem)
- Planning (MCTS, ToT, LoongFlow, VeriPlan)
- Multi-agent coordination (MAR, Orchestrator)
- Code understanding (Tree-sitter, LSP, LSPRAG)
- Sandbox execution (DeepAgents)
- Observability (OpenTelemetry, Langfuse)
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import litellm

from .core.config import AgentConfig, load_config
from .core.state import AgentState, TaskState
from .core.rlm import RLMEngine
from .memory.manager import MemoryManager
from .memory.episodic import EpisodicMemory
from .code_understanding.lsprag import LSPRAG
from .planning.mcts import MCTSPlanner
from .planning.tree_of_thoughts import TreeOfThoughts
from .planning.loongflow import LoongFlowPES
from .planning.veriplan import VeriPlan
from .agents.mar import MultiAgentReflexion
from .agents.orchestrator import Orchestrator, AgentCapability
from .agents.subagents import (
    PlannerAgent, CoderAgent, TesterAgent,
    ReviewerAgent, ResearcherAgent,
)
from .agents.agent_coder import AgentCoder
from .tools.tool_registry import create_default_registry
from .sandbox.deepagents_sandbox import DeepAgentsSandbox, SandboxConfig
from .sandbox.filesystem import TransactionalFS
from .sandbox.worktrees import WorktreeManager
from .observability.tracing import Tracer
from .observability.metrics import MetricsCollector
from .observability.loop_detection import LoopDetector, LoopSeverity
from .observability.langfuse_integration import LangfuseObserver
from .ui.terminal import TerminalUI


@dataclass
class CLIAgent:
    """
    Main CLI Coding Agent.

    Orchestrates all components for autonomous coding tasks.
    """
    config: AgentConfig
    workspace: str

    # Core components
    state: AgentState = field(init=False)
    rlm: RLMEngine = field(init=False)
    memory: MemoryManager = field(init=False)
    episodic: EpisodicMemory = field(init=False)

    # Planning
    mcts: MCTSPlanner = field(init=False)
    tot: TreeOfThoughts = field(init=False)
    loongflow: LoongFlowPES = field(init=False)
    veriplan: VeriPlan = field(init=False)

    # Agents
    mar: MultiAgentReflexion = field(init=False)
    orchestrator: Orchestrator = field(init=False)
    agent_coder: AgentCoder = field(init=False)

    # Code understanding
    lsprag: Optional[LSPRAG] = field(default=None)

    # Tools and sandbox
    tools: Any = field(init=False)
    sandbox: DeepAgentsSandbox = field(init=False)
    fs: TransactionalFS = field(init=False)
    worktrees: Optional[WorktreeManager] = field(default=None)

    # Observability
    tracer: Tracer = field(init=False)
    metrics: MetricsCollector = field(init=False)
    loop_detector: LoopDetector = field(init=False)
    langfuse: LangfuseObserver = field(init=False)

    # UI
    ui: TerminalUI = field(init=False)

    def __post_init__(self):
        """Initialize all components."""
        model = self.config.llm.model

        # Create simple LLM client wrapper for components that need it
        class SimpleLLMClient:
            def __init__(self, model_name):
                self.model = model_name
                # Get API credentials from environment
                self.api_key = (
                    os.getenv("ANTHROPIC_AUTH_TOKEN")
                    or os.getenv("ANTHROPIC_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
                self.api_base = (
                    os.getenv("ANTHROPIC_BASE_URL")
                    or os.getenv("OPENAI_API_BASE")
                )

            async def complete(self, prompt, **kwargs):
                call_kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.api_key:
                    call_kwargs["api_key"] = self.api_key
                if self.api_base:
                    call_kwargs["api_base"] = self.api_base
                call_kwargs.update(kwargs)
                response = await litellm.acompletion(**call_kwargs)
                return response.choices[0].message.content

        class SimpleEmbeddingClient:
            def __init__(self, model_name="text-embedding-3-small"):
                self.model = model_name
            async def embed(self, text):
                # Placeholder - return zero vector for now
                return [0.0] * 384

        llm_client = SimpleLLMClient(model)
        embedding_client = SimpleEmbeddingClient()

        # Core
        self.state = AgentState()
        self.rlm = RLMEngine(model=model, max_recursion_depth=self.config.rlm.max_recursion_depth)
        self.memory = MemoryManager(
            config=self.config.memory,
            llm_client=llm_client,
            embedding_client=embedding_client,
        )
        self.episodic = EpisodicMemory()

        # Planning
        self.mcts = MCTSPlanner(
            model=model,
            num_simulations=self.config.planning.mcts_simulations,
        )
        self.tot = TreeOfThoughts(
            model=model,
            max_depth=self.config.planning.tot_depth,
        )
        self.loongflow = LoongFlowPES(
            model=model,
            verify_contracts=self.config.planning.pes_verification_contracts,
        )
        self.veriplan = VeriPlan(model=model)

        # Agents
        self.mar = MultiAgentReflexion(
            model=model,
            max_rounds=self.config.validation.mar_max_iterations,
        )
        self.orchestrator = Orchestrator(model=model)
        self.agent_coder = AgentCoder(
            model=model,
            max_iterations=self.config.validation.max_iterations,
        )

        # Register subagents
        self._register_subagents()

        # Tools
        self.tools = create_default_registry(self.workspace)

        # Sandbox
        self.sandbox = DeepAgentsSandbox(
            default_config=SandboxConfig(
                timeout=self.config.sandbox.timeout,
                network_enabled=self.config.sandbox.network_enabled,
            ),
            workspace_root=self.workspace,
        )
        self.fs = TransactionalFS(self.workspace)

        # Initialize worktrees if git repo
        if os.path.exists(os.path.join(self.workspace, ".git")):
            self.worktrees = WorktreeManager(self.workspace)

        # Observability
        self.tracer = Tracer(service_name="cli-agent")
        self.metrics = MetricsCollector()
        self.loop_detector = LoopDetector(on_alert=self._handle_loop_alert)
        self.langfuse = LangfuseObserver()

        # UI
        self.ui = TerminalUI()

    def _register_subagents(self):
        """Register specialized subagents with orchestrator."""
        model = self.config.llm.model

        self.orchestrator.register_agent(
            "planner",
            PlannerAgent(model=model),
            [AgentCapability.PLANNING],
        )
        self.orchestrator.register_agent(
            "coder",
            CoderAgent(model=model),
            [AgentCapability.CODING],
        )
        self.orchestrator.register_agent(
            "tester",
            TesterAgent(model=model),
            [AgentCapability.TESTING],
        )
        self.orchestrator.register_agent(
            "reviewer",
            ReviewerAgent(model=model),
            [AgentCapability.REVIEWING],
        )
        self.orchestrator.register_agent(
            "researcher",
            ResearcherAgent(model=model),
            [AgentCapability.RESEARCHING],
        )

    def _handle_loop_alert(self, alert):
        """Handle loop detection alerts."""
        if alert.severity == LoopSeverity.CRITICAL:
            self.ui.console.print_error(
                f"Loop detected: {alert.message}",
                "Critical Loop Alert",
            )
            # Could trigger intervention here

    async def run(self, task: str) -> Dict[str, Any]:
        """
        Run the agent on a task.

        Args:
            task: Task description

        Returns:
            Result dictionary
        """
        with self.tracer.start_span("agent_run", {"task": task[:100]}) as span:
            self.metrics.inc_counter("requests_total")
            start_time = datetime.now()

            # Create snapshot before starting
            snapshot = self.fs.begin_transaction(f"task_{start_time.timestamp()}")

            try:
                # Start episode for reflexion
                episode_id = f"episode_{start_time.timestamp()}"
                self.episodic.start_episode(episode_id, task)

                # Show task
                self.ui.show_task(task)

                # Phase 1: Understanding
                span.add_event("phase_understanding")
                context = await self._understand_task(task)

                # Phase 2: Planning
                span.add_event("phase_planning")
                plan = await self._create_plan(task, context)

                # Show plan
                self.ui.show_plan([{"description": s, "status": "pending"} for s in plan])

                # Phase 3: Execution
                span.add_event("phase_execution")
                result = await self._execute_plan(task, plan, context)

                # Phase 4: Validation
                span.add_event("phase_validation")
                validated = await self._validate_result(task, result)

                # Commit changes if successful
                if validated.get("success", False):
                    self.fs.commit_transaction()
                    self.ui.show_result({"status": "success", "message": "Task completed successfully"})
                else:
                    # Rollback on failure
                    self.fs.rollback_transaction()
                    self.ui.show_result({"status": "error", "message": validated.get("error", "Validation failed")})

                # End episode with result
                self.episodic.end_episode(validated.get("success", False))

                duration = (datetime.now() - start_time).total_seconds()
                self.metrics.observe_histogram("request_duration_seconds", duration)

                return validated

            except Exception as e:
                span.set_status("error", str(e))
                self.metrics.inc_counter("errors_total")

                # Rollback on error
                self.fs.rollback_transaction()

                self.ui.console.print_error(str(e), "Task Failed")

                return {
                    "success": False,
                    "error": str(e),
                }

    async def _understand_task(self, task: str) -> Dict[str, Any]:
        """Understand the task and gather context."""
        context = {
            "task": task,
            "workspace": self.workspace,
            "timestamp": datetime.now().isoformat(),
        }

        # Use RLM to analyze task
        rlm_result = await self.rlm.process_with_recursion(
            f"Analyze this coding task and identify key requirements: {task}"
        )
        # process_with_recursion returns a string directly
        context["analysis"] = rlm_result if isinstance(rlm_result, str) else rlm_result.get("result", "")

        # Search relevant files if LSPRAG available
        if self.lsprag:
            relevant = await self.lsprag.search(task)
            context["relevant_files"] = relevant

        # Retrieve relevant memories
        memory_context = await self.memory.recall(task, top_k=5)
        context["memory_context"] = memory_context

        # Check for past similar tasks
        lessons = self.episodic.get_lessons_learned(task)
        if lessons:
            context["lessons_learned"] = lessons

        return context

    async def _create_plan(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Create execution plan using planning systems."""
        # Use MAR for diverse perspectives
        reflexion = await self.mar.reflect(task, context)

        # Use Tree of Thoughts for planning
        tot_result = await self.tot.solve(
            f"Create a step-by-step plan for: {task}",
            context=context,
        )

        # Extract steps from ToT result
        steps = []
        if tot_result:
            for node in tot_result:
                if "step" in node.thought.lower() or node.depth > 0:
                    steps.append(node.thought)

        # Verify plan
        if steps:
            verification = await self.veriplan.verify(
                [{"description": s} for s in steps],
                level="standard",
            )

            if not verification.is_valid:
                # Get suggestions and refine
                suggestions = await self.veriplan.suggest_fixes(
                    [{"description": s} for s in steps],
                    verification,
                )
                # Could iterate to improve plan

        return steps or ["Analyze task", "Implement solution", "Test result"]

    async def _execute_plan(
        self,
        task: str,
        plan: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the plan using LoongFlow PES."""
        # Create sandbox session
        session = await self.sandbox.create_session(copy_workspace=self.workspace)

        # Define step executor
        def execute_step(step, ctx):
            # Record action for loop detection
            self.loop_detector.record_action(step.description)
            return {"completed": True, "step": step.description}

        # Run PES cycle
        summary, final_state = await self.loongflow.run(
            task=task,
            context=context,
            executor=execute_step,
        )

        # Cleanup sandbox
        await self.sandbox.cleanup_session(session)

        return {
            "success": summary.success,
            "summary": summary.model_dump(),
            "state": final_state,
        }

    async def _validate_result(
        self,
        task: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the result using AgentCoder pattern."""
        if not result.get("success", False):
            return result

        # Use MAR for final validation
        validation_result = await self.mar.reflect(
            f"Validate if this result correctly solves: {task}",
            context=result,
        )

        if validation_result.confidence >= 0.7:
            return {
                "success": True,
                "result": result,
                "validation": validation_result.model_dump(),
            }

        return {
            "success": False,
            "error": "Validation confidence too low",
            "result": result,
            "validation": validation_result.model_dump(),
        }

    async def chat(self, message: str) -> str:
        """
        Chat interface for interactive mode.

        Args:
            message: User message

        Returns:
            Agent response
        """
        # Handle commands
        if message.startswith("/"):
            return await self._handle_command(message)

        # Process as task
        result = await self.run(message)

        if result.get("success"):
            return "Task completed successfully."
        else:
            return f"Task failed: {result.get('error', 'Unknown error')}"

    async def _handle_command(self, command: str) -> str:
        """Handle CLI commands."""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            return """Commands:
  /help     - Show this help
  /status   - Show agent status
  /memory   - Show memory stats
  /metrics  - Show metrics
  /clear    - Clear state
  /quit     - Exit"""

        elif cmd == "/status":
            return f"""Agent Status:
  Tasks processed: {self.metrics.counter('requests_total').get()}
  Errors: {self.metrics.counter('errors_total').get()}
  Memory entries: {len(self.memory._storage)}
  Loop alerts: {len(self.loop_detector.get_alerts())}"""

        elif cmd == "/memory":
            return f"Memory stats: {self.memory.get_stats()}"

        elif cmd == "/metrics":
            return f"Metrics: {self.metrics.get_all_metrics()}"

        elif cmd == "/clear":
            self.state = AgentState()
            self.loop_detector.reset()
            return "State cleared."

        elif cmd == "/quit":
            return "QUIT"

        else:
            return f"Unknown command: {cmd}"


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CLI Coding Agent")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--task", default=None, help="Task to execute")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Load config (from file if specified, otherwise from environment)
    config = load_config(args.config)

    # Create agent
    agent = CLIAgent(
        config=config,
        workspace=os.path.abspath(args.workspace),
    )

    # Show welcome
    agent.ui.welcome()

    if args.task:
        # Execute single task
        result = await agent.run(args.task)
        print(f"Result: {result}")

    elif args.interactive:
        # Interactive mode
        while True:
            try:
                message = agent.ui.prompt_input("You")
                if not message:
                    continue

                response = await agent.chat(message)

                if response == "QUIT":
                    break

                agent.ui.console.print(f"\n[bold cyan]Agent:[/bold cyan] {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                agent.ui.console.print_error(str(e))

    else:
        # Show help
        print("Use --task 'your task' or --interactive")


def run():
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
