"""
LoongFlow PES Paradigm - Plan/Execute/Summary with Verification Contracts

Based on research for structured code generation:
- Plan: Generate verified execution plan
- Execute: Run plan with monitoring
- Summary: Analyze results and update knowledge
- Verification contracts: Pre/post conditions for each step
- Supports hierarchical decomposition
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime
import litellm
from pydantic import BaseModel, Field


class PlanStepType(str, Enum):
    """Types of plan steps."""
    ANALYZE = "analyze"        # Analyze requirements/code
    DESIGN = "design"          # Design solution approach
    IMPLEMENT = "implement"    # Write code
    TEST = "test"              # Run tests
    VALIDATE = "validate"      # Validate results
    REFACTOR = "refactor"      # Improve code
    DOCUMENT = "document"      # Add documentation
    DEPLOY = "deploy"          # Deploy changes


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ContractType(str, Enum):
    """Types of verification contracts."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"


@dataclass
class PESContract:
    """
    Verification contract for plan step.

    Defines pre/post conditions that must hold
    for a step to be valid.
    """
    contract_type: ContractType
    description: str
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    llm_check: bool = True  # Use LLM to verify if no check_fn

    # Verification results
    verified: bool = False
    verification_time: Optional[datetime] = None
    verification_notes: str = ""

    async def verify(
        self,
        state: Dict[str, Any],
        model: str = "gpt-4",
    ) -> bool:
        """Verify the contract against current state."""
        if self.check_fn:
            self.verified = self.check_fn(state)
        elif self.llm_check:
            self.verified = await self._llm_verify(state, model)
        else:
            self.verified = True  # No verification specified

        self.verification_time = datetime.now()
        return self.verified

    async def _llm_verify(
        self,
        state: Dict[str, Any],
        model: str,
    ) -> bool:
        """Use LLM to verify contract."""
        prompt = f"""Verify if this {self.contract_type.value} holds:

Condition: {self.description}

Current state:
{self._format_state(state)}

Does this condition hold? Answer YES or NO with brief explanation."""

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        self.verification_notes = answer
        return "YES" in answer.upper()

    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for prompt."""
        lines = []
        for key, value in state.items():
            if isinstance(value, str) and len(value) > 200:
                lines.append(f"  {key}: {value[:200]}...")
            elif isinstance(value, (list, dict)):
                lines.append(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class PlanStep:
    """
    Single step in execution plan.

    Includes verification contracts and
    execution metadata.
    """
    id: str
    step_type: PlanStepType
    description: str
    details: str = ""

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Verification contracts
    preconditions: List[PESContract] = field(default_factory=list)
    postconditions: List[PESContract] = field(default_factory=list)
    invariants: List[PESContract] = field(default_factory=list)

    # Execution state
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    # Sub-steps for hierarchical decomposition
    substeps: List["PlanStep"] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf step (no substeps)."""
        return len(self.substeps) == 0

    def add_substep(self, substep: "PlanStep") -> None:
        """Add a substep."""
        substep.parent_id = self.id
        self.substeps.append(substep)

    def add_precondition(self, description: str, check_fn: Optional[Callable] = None) -> None:
        """Add a precondition contract."""
        self.preconditions.append(PESContract(
            contract_type=ContractType.PRECONDITION,
            description=description,
            check_fn=check_fn,
        ))

    def add_postcondition(self, description: str, check_fn: Optional[Callable] = None) -> None:
        """Add a postcondition contract."""
        self.postconditions.append(PESContract(
            contract_type=ContractType.POSTCONDITION,
            description=description,
            check_fn=check_fn,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "description": self.description,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "num_substeps": len(self.substeps),
            "duration": self.duration,
            "error": self.error,
        }


class ExecutionSummary(BaseModel):
    """Summary of plan execution."""
    task: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    total_duration: float
    success: bool
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    changes_made: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)


class LoongFlowPES:
    """
    LoongFlow PES (Plan/Execute/Summary) Paradigm.

    Implements structured code generation with:
    - Hierarchical planning with verification contracts
    - Monitored execution with rollback support
    - Comprehensive summarization for learning
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_plan_depth: int = 3,
        max_retries: int = 3,
        verify_contracts: bool = True,
        auto_decompose: bool = True,
    ):
        self.model = model
        self.max_plan_depth = max_plan_depth
        self.max_retries = max_retries
        self.verify_contracts = verify_contracts
        self.auto_decompose = auto_decompose

        self._step_counter = 0
        self._execution_history: List[Dict[str, Any]] = []

    def _generate_step_id(self) -> str:
        """Generate unique step ID."""
        self._step_counter += 1
        return f"step_{self._step_counter}"

    async def run(
        self,
        task: str,
        context: Dict[str, Any],
        executor: Callable[[PlanStep, Dict[str, Any]], Any],
    ) -> Tuple[ExecutionSummary, Dict[str, Any]]:
        """
        Run the complete PES cycle.

        Args:
            task: Task description
            context: Initial context/state
            executor: Function to execute individual steps

        Returns:
            Tuple of (execution summary, final state)
        """
        start_time = datetime.now()

        # PLAN Phase
        plan = await self.plan(task, context)

        # EXECUTE Phase
        final_state = await self.execute(plan, context, executor)

        # SUMMARY Phase
        summary = await self.summarize(
            task, plan, context, final_state, start_time
        )

        return summary, final_state

    async def plan(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[PlanStep]:
        """
        PLAN Phase: Generate verified execution plan.

        Creates hierarchical plan with verification contracts
        for each step.
        """
        # Generate initial plan
        prompt = f"""Create an execution plan for this task.

Task: {task}

Context:
{self._format_context(context)}

Generate a structured plan with steps. For each step specify:
1. Type: one of [analyze, design, implement, test, validate, refactor, document, deploy]
2. Description: what needs to be done
3. Details: specific instructions
4. Dependencies: which previous steps must complete first (by step number)
5. Preconditions: what must be true before this step
6. Postconditions: what must be true after this step

Format each step as:
STEP <number>:
TYPE: <type>
DESCRIPTION: <description>
DETAILS: <details>
DEPENDS_ON: <comma-separated step numbers, or "none">
PRECONDITIONS: <conditions>
POSTCONDITIONS: <conditions>

Plan:"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse plan
        steps = self._parse_plan(response.choices[0].message.content)

        # Auto-decompose complex steps if enabled
        if self.auto_decompose:
            steps = await self._decompose_complex_steps(steps, task, context)

        return steps

    async def execute(
        self,
        plan: List[PlanStep],
        context: Dict[str, Any],
        executor: Callable[[PlanStep, Dict[str, Any]], Any],
    ) -> Dict[str, Any]:
        """
        EXECUTE Phase: Run plan with monitoring.

        Executes each step while verifying contracts
        and handling failures.
        """
        current_state = context.copy()
        completed_steps: Dict[str, PlanStep] = {}

        for step in plan:
            # Check dependencies
            if not self._dependencies_met(step, completed_steps):
                step.status = StepStatus.BLOCKED
                continue

            # Verify preconditions
            if self.verify_contracts:
                preconditions_ok = await self._verify_contracts(
                    step.preconditions, current_state
                )
                if not preconditions_ok:
                    step.status = StepStatus.FAILED
                    step.error = "Preconditions not met"
                    self._record_execution(step, current_state, success=False)
                    continue

            # Execute step with retries
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now()

            success = False
            for attempt in range(self.max_retries):
                try:
                    # Execute leaf step or recursively execute substeps
                    if step.is_leaf:
                        result = await self._execute_step(
                            step, current_state, executor
                        )
                    else:
                        result = await self._execute_substeps(
                            step, current_state, executor
                        )

                    step.result = result
                    current_state = self._update_state(current_state, step, result)

                    # Verify postconditions
                    if self.verify_contracts:
                        postconditions_ok = await self._verify_contracts(
                            step.postconditions, current_state
                        )
                        if not postconditions_ok:
                            raise ValueError("Postconditions not met")

                    success = True
                    break

                except Exception as e:
                    step.error = str(e)
                    if attempt < self.max_retries - 1:
                        # Retry with reflection
                        await self._reflect_on_failure(step, current_state, e)

            step.completed_at = datetime.now()
            step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            self._record_execution(step, current_state, success)

            if success:
                completed_steps[step.id] = step

        return current_state

    async def summarize(
        self,
        task: str,
        plan: List[PlanStep],
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any],
        start_time: datetime,
    ) -> ExecutionSummary:
        """
        SUMMARY Phase: Analyze execution and extract insights.

        Generates comprehensive summary including:
        - Execution statistics
        - Insights and recommendations
        - Lessons learned for future tasks
        """
        total_steps = len(plan)
        completed_steps = sum(1 for s in plan if s.status == StepStatus.COMPLETED)
        failed_steps = sum(1 for s in plan if s.status == StepStatus.FAILED)
        skipped_steps = sum(
            1 for s in plan
            if s.status in (StepStatus.SKIPPED, StepStatus.BLOCKED)
        )

        total_duration = (datetime.now() - start_time).total_seconds()
        success = failed_steps == 0 and completed_steps > 0

        # Generate insights via LLM
        insights_prompt = f"""Analyze this task execution and provide insights.

Task: {task}

Execution summary:
- Total steps: {total_steps}
- Completed: {completed_steps}
- Failed: {failed_steps}
- Skipped: {skipped_steps}
- Duration: {total_duration:.2f}s
- Success: {success}

Steps executed:
{self._format_steps_for_summary(plan)}

Initial state keys: {list(initial_state.keys())}
Final state keys: {list(final_state.keys())}

Provide:
1. INSIGHTS: Key observations about the execution (2-3 points)
2. RECOMMENDATIONS: Improvements for similar tasks (2-3 points)
3. CHANGES: What was actually changed/created (list)
4. LESSONS: What to remember for future (2-3 points)

Format:
INSIGHTS:
- <insight 1>
- <insight 2>

RECOMMENDATIONS:
- <recommendation 1>
- <recommendation 2>

CHANGES:
- <change 1>
- <change 2>

LESSONS:
- <lesson 1>
- <lesson 2>"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": insights_prompt}],
            temperature=0.3,
        )

        # Parse insights
        parsed = self._parse_summary_response(response.choices[0].message.content)

        return ExecutionSummary(
            task=task,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            total_duration=total_duration,
            success=success,
            insights=parsed.get("insights", []),
            recommendations=parsed.get("recommendations", []),
            changes_made=parsed.get("changes", []),
            lessons_learned=parsed.get("lessons", []),
        )

    async def _execute_step(
        self,
        step: PlanStep,
        state: Dict[str, Any],
        executor: Callable[[PlanStep, Dict[str, Any]], Any],
    ) -> Any:
        """Execute a single step."""
        return await asyncio.to_thread(executor, step, state)

    async def _execute_substeps(
        self,
        step: PlanStep,
        state: Dict[str, Any],
        executor: Callable[[PlanStep, Dict[str, Any]], Any],
    ) -> Any:
        """Recursively execute substeps."""
        current_state = state.copy()

        for substep in step.substeps:
            substep.status = StepStatus.IN_PROGRESS
            substep.started_at = datetime.now()

            try:
                if substep.is_leaf:
                    result = await self._execute_step(substep, current_state, executor)
                else:
                    result = await self._execute_substeps(substep, current_state, executor)

                substep.result = result
                current_state = self._update_state(current_state, substep, result)
                substep.status = StepStatus.COMPLETED

            except Exception as e:
                substep.status = StepStatus.FAILED
                substep.error = str(e)
                raise

            finally:
                substep.completed_at = datetime.now()

        return current_state

    async def _decompose_complex_steps(
        self,
        steps: List[PlanStep],
        task: str,
        context: Dict[str, Any],
    ) -> List[PlanStep]:
        """Decompose complex steps into substeps."""
        for step in steps:
            if self._is_complex_step(step):
                substeps = await self._generate_substeps(step, task, context)
                for substep in substeps:
                    step.add_substep(substep)

        return steps

    def _is_complex_step(self, step: PlanStep) -> bool:
        """Check if a step needs decomposition."""
        # Heuristics for complexity
        indicators = [
            len(step.details) > 200,
            "multiple" in step.description.lower(),
            "several" in step.description.lower(),
            step.step_type in (PlanStepType.IMPLEMENT, PlanStepType.REFACTOR),
        ]
        return sum(indicators) >= 2

    async def _generate_substeps(
        self,
        step: PlanStep,
        task: str,
        context: Dict[str, Any],
    ) -> List[PlanStep]:
        """Generate substeps for a complex step."""
        prompt = f"""Decompose this step into smaller substeps.

Parent task: {task}
Step: {step.description}
Details: {step.details}

Break this into 2-4 smaller, concrete substeps.
Format each substep as:
SUBSTEP <number>: <description>

Substeps:"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        substeps = []
        lines = response.choices[0].message.content.strip().split("\n")

        for line in lines:
            if line.strip().upper().startswith("SUBSTEP"):
                desc = line.split(":", 1)[1].strip() if ":" in line else line
                substeps.append(PlanStep(
                    id=self._generate_step_id(),
                    step_type=step.step_type,
                    description=desc,
                ))

        return substeps

    async def _verify_contracts(
        self,
        contracts: List[PESContract],
        state: Dict[str, Any],
    ) -> bool:
        """Verify all contracts."""
        for contract in contracts:
            if not await contract.verify(state, self.model):
                return False
        return True

    async def _reflect_on_failure(
        self,
        step: PlanStep,
        state: Dict[str, Any],
        error: Exception,
    ) -> None:
        """Reflect on step failure for retry."""
        prompt = f"""A step failed during execution. Analyze the failure.

Step: {step.description}
Details: {step.details}
Error: {str(error)}

What might have caused this failure and how should the retry differ?"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Store reflection for learning
        step.details += f"\n\nRetry insight: {response.choices[0].message.content}"

    def _dependencies_met(
        self,
        step: PlanStep,
        completed: Dict[str, PlanStep],
    ) -> bool:
        """Check if step dependencies are met."""
        for dep_id in step.depends_on:
            if dep_id not in completed:
                return False
            if completed[dep_id].status != StepStatus.COMPLETED:
                return False
        return True

    def _update_state(
        self,
        state: Dict[str, Any],
        step: PlanStep,
        result: Any,
    ) -> Dict[str, Any]:
        """Update state after step execution."""
        new_state = state.copy()
        new_state[f"step_{step.id}_result"] = result
        new_state[f"step_{step.id}_completed"] = True
        new_state["last_completed_step"] = step.id
        return new_state

    def _record_execution(
        self,
        step: PlanStep,
        state: Dict[str, Any],
        success: bool,
    ) -> None:
        """Record step execution in history."""
        self._execution_history.append({
            "step_id": step.id,
            "step_type": step.step_type.value,
            "description": step.description,
            "success": success,
            "duration": step.duration,
            "timestamp": datetime.now().isoformat(),
        })

    def _parse_plan(self, text: str) -> List[PlanStep]:
        """Parse plan from LLM response."""
        steps = []
        current_step = None
        step_map: Dict[int, str] = {}  # Map step numbers to IDs

        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("STEP"):
                # Save previous step
                if current_step:
                    steps.append(current_step)

                # Start new step
                step_id = self._generate_step_id()
                step_num = len(steps) + 1
                step_map[step_num] = step_id

                current_step = PlanStep(
                    id=step_id,
                    step_type=PlanStepType.IMPLEMENT,  # Default
                    description="",
                )

            elif current_step:
                if upper.startswith("TYPE:"):
                    type_str = line.split(":", 1)[1].strip().lower()
                    try:
                        current_step.step_type = PlanStepType(type_str)
                    except ValueError:
                        pass

                elif upper.startswith("DESCRIPTION:"):
                    current_step.description = line.split(":", 1)[1].strip()

                elif upper.startswith("DETAILS:"):
                    current_step.details = line.split(":", 1)[1].strip()

                elif upper.startswith("DEPENDS_ON:"):
                    deps_str = line.split(":", 1)[1].strip().lower()
                    if deps_str != "none":
                        # Parse dependency numbers and convert to IDs
                        import re
                        dep_nums = re.findall(r"\d+", deps_str)
                        for num_str in dep_nums:
                            num = int(num_str)
                            if num in step_map:
                                current_step.depends_on.append(step_map[num])

                elif upper.startswith("PRECONDITIONS:"):
                    cond = line.split(":", 1)[1].strip()
                    if cond.lower() not in ("none", ""):
                        current_step.add_precondition(cond)

                elif upper.startswith("POSTCONDITIONS:"):
                    cond = line.split(":", 1)[1].strip()
                    if cond.lower() not in ("none", ""):
                        current_step.add_postcondition(cond)

        # Add last step
        if current_step:
            steps.append(current_step)

        return steps

    def _parse_summary_response(self, text: str) -> Dict[str, List[str]]:
        """Parse summary response from LLM."""
        result = {
            "insights": [],
            "recommendations": [],
            "changes": [],
            "lessons": [],
        }

        current_section = None
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            upper = line.upper()

            if "INSIGHTS:" in upper:
                current_section = "insights"
            elif "RECOMMENDATIONS:" in upper:
                current_section = "recommendations"
            elif "CHANGES:" in upper:
                current_section = "changes"
            elif "LESSONS:" in upper:
                current_section = "lessons"
            elif line.startswith("-") and current_section:
                item = line[1:].strip()
                if item:
                    result[current_section].append(item)

        return result

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts."""
        lines = []
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 100:
                lines.append(f"  {key}: {value[:100]}...")
            elif isinstance(value, (list, dict)):
                lines.append(f"  {key}: {type(value).__name__} ({len(value)} items)")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _format_steps_for_summary(self, steps: List[PlanStep]) -> str:
        """Format steps for summary prompt."""
        lines = []
        for step in steps:
            status_icon = {
                StepStatus.COMPLETED: "[OK]",
                StepStatus.FAILED: "[FAIL]",
                StepStatus.SKIPPED: "[SKIP]",
                StepStatus.BLOCKED: "[BLOCK]",
            }.get(step.status, "[?]")

            duration = f" ({step.duration:.2f}s)" if step.duration else ""
            error = f" - Error: {step.error}" if step.error else ""

            lines.append(f"{status_icon} {step.description}{duration}{error}")

        return "\n".join(lines)

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history.copy()
