"""
VeriPlan - Formal Verification Layer for Plans

Based on research for verified code generation:
- Formal specification of plan properties
- Constraint satisfaction checking
- Dependency graph validation
- Resource conflict detection
- Deadlock and cycle detection
- Safety invariant verification
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import litellm
from pydantic import BaseModel


class VerificationLevel(str, Enum):
    """Levels of verification thoroughness."""
    QUICK = "quick"          # Basic structural checks
    STANDARD = "standard"    # Standard verification
    THOROUGH = "thorough"    # Comprehensive with LLM validation


class VerificationStatus(str, Enum):
    """Status of verification check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ViolationType(str, Enum):
    """Types of verification violations."""
    DEPENDENCY_CYCLE = "dependency_cycle"
    MISSING_DEPENDENCY = "missing_dependency"
    RESOURCE_CONFLICT = "resource_conflict"
    INVALID_PRECONDITION = "invalid_precondition"
    UNREACHABLE_STEP = "unreachable_step"
    DEADLOCK = "deadlock"
    SAFETY_VIOLATION = "safety_violation"
    INCOMPLETE_PLAN = "incomplete_plan"
    ORDERING_VIOLATION = "ordering_violation"


@dataclass
class VerificationViolation:
    """A single verification violation."""
    violation_type: ViolationType
    severity: str  # "error", "warning", "info"
    message: str
    affected_steps: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.violation_type.value,
            "severity": self.severity,
            "message": self.message,
            "affected_steps": self.affected_steps,
            "suggestion": self.suggestion,
        }


@dataclass
class VerificationResult:
    """Result of plan verification."""
    status: VerificationStatus
    violations: List[VerificationViolation] = field(default_factory=list)
    warnings: List[VerificationViolation] = field(default_factory=list)
    checks_performed: List[str] = field(default_factory=list)
    verification_time: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if plan passed verification."""
        return self.status in (VerificationStatus.PASSED, VerificationStatus.WARNING)

    @property
    def error_count(self) -> int:
        """Count of error-level violations."""
        return len([v for v in self.violations if v.severity == "error"])

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def add_violation(self, violation: VerificationViolation) -> None:
        """Add a violation."""
        if violation.severity == "warning":
            self.warnings.append(violation)
        else:
            self.violations.append(violation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [v.to_dict() for v in self.warnings],
            "checks_performed": self.checks_performed,
            "verification_time": self.verification_time,
        }


class PlanSpec(BaseModel):
    """Formal specification for a plan."""
    task_goal: str
    required_outputs: List[str] = []
    forbidden_actions: List[str] = []
    required_actions: List[str] = []
    resource_constraints: Dict[str, int] = {}  # resource -> max concurrent usage
    ordering_constraints: List[Tuple[str, str]] = []  # (before, after) pairs
    safety_invariants: List[str] = []
    timeout_seconds: Optional[float] = None


class VeriPlan:
    """
    Formal verification layer for execution plans.

    Performs multi-level verification including:
    - Structural analysis (cycles, reachability)
    - Resource conflict detection
    - Safety invariant checking
    - LLM-based semantic validation
    """

    def __init__(
        self,
        model: str = "gpt-4",
        default_level: VerificationLevel = VerificationLevel.STANDARD,
    ):
        self.model = model
        self.default_level = default_level

    async def verify(
        self,
        plan: List[Any],  # List of PlanStep
        spec: Optional[PlanSpec] = None,
        level: Optional[VerificationLevel] = None,
    ) -> VerificationResult:
        """
        Verify a plan against specifications.

        Args:
            plan: List of plan steps
            spec: Optional formal specification
            level: Verification thoroughness level

        Returns:
            VerificationResult with all findings
        """
        import time
        start_time = time.time()

        level = level or self.default_level
        result = VerificationResult(status=VerificationStatus.PASSED)

        # Build dependency graph
        dep_graph = self._build_dependency_graph(plan)

        # Structural verification (always performed)
        await self._verify_structure(plan, dep_graph, result)
        result.checks_performed.append("structural_analysis")

        # Dependency verification
        await self._verify_dependencies(plan, dep_graph, result)
        result.checks_performed.append("dependency_verification")

        # Reachability analysis
        await self._verify_reachability(plan, dep_graph, result)
        result.checks_performed.append("reachability_analysis")

        if level in (VerificationLevel.STANDARD, VerificationLevel.THOROUGH):
            # Resource conflict detection
            await self._verify_resources(plan, spec, result)
            result.checks_performed.append("resource_verification")

            # Ordering constraint verification
            if spec and spec.ordering_constraints:
                await self._verify_ordering(plan, spec, result)
                result.checks_performed.append("ordering_verification")

        if level == VerificationLevel.THOROUGH:
            # Safety invariant verification
            if spec and spec.safety_invariants:
                await self._verify_safety(plan, spec, result)
                result.checks_performed.append("safety_verification")

            # Semantic verification via LLM
            await self._verify_semantic(plan, spec, result)
            result.checks_performed.append("semantic_verification")

            # Completeness check
            if spec:
                await self._verify_completeness(plan, spec, result)
                result.checks_performed.append("completeness_verification")

        # Determine final status
        if result.violations:
            result.status = VerificationStatus.FAILED
        elif result.warnings:
            result.status = VerificationStatus.WARNING

        result.verification_time = time.time() - start_time
        return result

    def _build_dependency_graph(
        self,
        plan: List[Any],
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from plan."""
        graph: Dict[str, Set[str]] = {}

        for step in plan:
            step_id = step.id if hasattr(step, 'id') else str(step)
            deps = step.depends_on if hasattr(step, 'depends_on') else []
            graph[step_id] = set(deps)

        return graph

    async def _verify_structure(
        self,
        plan: List[Any],
        dep_graph: Dict[str, Set[str]],
        result: VerificationResult,
    ) -> None:
        """Verify plan structure."""
        # Check for empty plan
        if not plan:
            result.add_violation(VerificationViolation(
                violation_type=ViolationType.INCOMPLETE_PLAN,
                severity="error",
                message="Plan is empty",
                suggestion="Generate at least one plan step",
            ))
            return

        # Check for cycles using DFS
        cycles = self._detect_cycles(dep_graph)
        for cycle in cycles:
            result.add_violation(VerificationViolation(
                violation_type=ViolationType.DEPENDENCY_CYCLE,
                severity="error",
                message=f"Dependency cycle detected: {' -> '.join(cycle)}",
                affected_steps=cycle,
                suggestion="Remove circular dependencies",
            ))

    async def _verify_dependencies(
        self,
        plan: List[Any],
        dep_graph: Dict[str, Set[str]],
        result: VerificationResult,
    ) -> None:
        """Verify dependency relationships."""
        step_ids = set(dep_graph.keys())

        for step_id, deps in dep_graph.items():
            for dep in deps:
                if dep not in step_ids:
                    result.add_violation(VerificationViolation(
                        violation_type=ViolationType.MISSING_DEPENDENCY,
                        severity="error",
                        message=f"Step '{step_id}' depends on non-existent step '{dep}'",
                        affected_steps=[step_id, dep],
                        suggestion=f"Add step '{dep}' or remove dependency",
                    ))

    async def _verify_reachability(
        self,
        plan: List[Any],
        dep_graph: Dict[str, Set[str]],
        result: VerificationResult,
    ) -> None:
        """Verify all steps are reachable."""
        # Find root steps (no dependencies)
        roots = [s for s, deps in dep_graph.items() if not deps]

        if not roots:
            result.add_violation(VerificationViolation(
                violation_type=ViolationType.DEADLOCK,
                severity="error",
                message="No root steps found - all steps have dependencies",
                suggestion="Ensure at least one step has no dependencies",
            ))
            return

        # BFS from roots to find reachable steps
        reachable: Set[str] = set()
        # Build reverse graph
        reverse_graph: Dict[str, Set[str]] = {s: set() for s in dep_graph}
        for step_id, deps in dep_graph.items():
            for dep in deps:
                if dep in reverse_graph:
                    reverse_graph[dep].add(step_id)

        # BFS
        queue = list(roots)
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            queue.extend(reverse_graph.get(current, []))

        # Check for unreachable steps
        unreachable = set(dep_graph.keys()) - reachable
        for step_id in unreachable:
            result.add_violation(VerificationViolation(
                violation_type=ViolationType.UNREACHABLE_STEP,
                severity="warning",
                message=f"Step '{step_id}' is unreachable from root steps",
                affected_steps=[step_id],
                suggestion="Check dependency configuration",
            ))

    async def _verify_resources(
        self,
        plan: List[Any],
        spec: Optional[PlanSpec],
        result: VerificationResult,
    ) -> None:
        """Verify resource constraints."""
        if not spec or not spec.resource_constraints:
            return

        # Track resource usage per step
        resource_usage: Dict[str, List[str]] = {}

        for step in plan:
            step_id = step.id if hasattr(step, 'id') else str(step)
            # Extract resources from step (simplified - would need proper parsing)
            step_desc = str(step.description if hasattr(step, 'description') else step)

            for resource, max_usage in spec.resource_constraints.items():
                if resource.lower() in step_desc.lower():
                    if resource not in resource_usage:
                        resource_usage[resource] = []
                    resource_usage[resource].append(step_id)

        # Check for conflicts (simplified - would need proper scheduling)
        for resource, steps in resource_usage.items():
            max_allowed = spec.resource_constraints.get(resource, 1)
            if len(steps) > max_allowed:
                result.add_violation(VerificationViolation(
                    violation_type=ViolationType.RESOURCE_CONFLICT,
                    severity="warning",
                    message=f"Resource '{resource}' may be over-allocated: {len(steps)} steps vs max {max_allowed}",
                    affected_steps=steps,
                    suggestion="Consider sequencing steps that use this resource",
                ))

    async def _verify_ordering(
        self,
        plan: List[Any],
        spec: PlanSpec,
        result: VerificationResult,
    ) -> None:
        """Verify ordering constraints."""
        step_order = {
            (step.id if hasattr(step, 'id') else str(step)): i
            for i, step in enumerate(plan)
        }

        for before, after in spec.ordering_constraints:
            if before in step_order and after in step_order:
                if step_order[before] >= step_order[after]:
                    result.add_violation(VerificationViolation(
                        violation_type=ViolationType.ORDERING_VIOLATION,
                        severity="error",
                        message=f"Ordering constraint violated: '{before}' must come before '{after}'",
                        affected_steps=[before, after],
                        suggestion=f"Reorder steps so '{before}' precedes '{after}'",
                    ))

    async def _verify_safety(
        self,
        plan: List[Any],
        spec: PlanSpec,
        result: VerificationResult,
    ) -> None:
        """Verify safety invariants."""
        plan_text = "\n".join(
            str(step.description if hasattr(step, 'description') else step)
            for step in plan
        )

        for invariant in spec.safety_invariants:
            prompt = f"""Analyze if this plan maintains the safety invariant.

Safety Invariant: {invariant}

Plan:
{plan_text}

Does this plan potentially violate the safety invariant at any point?
Answer YES if it might violate, NO if it's safe.
Explain briefly."""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            answer = response.choices[0].message.content.strip()
            if "YES" in answer.upper().split()[0] if answer else False:
                result.add_violation(VerificationViolation(
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity="error",
                    message=f"Safety invariant may be violated: {invariant}",
                    suggestion=answer,
                ))

    async def _verify_semantic(
        self,
        plan: List[Any],
        spec: Optional[PlanSpec],
        result: VerificationResult,
    ) -> None:
        """Verify semantic correctness via LLM."""
        plan_text = "\n".join(
            f"- {step.description if hasattr(step, 'description') else step}"
            for step in plan
        )

        goal = spec.task_goal if spec else "complete the task"
        forbidden = spec.forbidden_actions if spec else []

        prompt = f"""Analyze this execution plan for potential issues.

Goal: {goal}

Plan:
{plan_text}

{"Forbidden actions: " + ", ".join(forbidden) if forbidden else ""}

Identify any potential issues:
1. Missing steps that are necessary
2. Steps that may cause problems
3. Logical inconsistencies
4. Security or safety concerns

Format issues as:
ISSUE: <description>
SEVERITY: <error/warning/info>
SUGGESTION: <how to fix>

If no issues found, respond with "NO ISSUES FOUND"."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        # Parse issues
        text = response.choices[0].message.content.strip()
        if "NO ISSUES FOUND" not in text.upper():
            issues = self._parse_issues(text)
            for issue in issues:
                result.add_violation(issue)

    async def _verify_completeness(
        self,
        plan: List[Any],
        spec: PlanSpec,
        result: VerificationResult,
    ) -> None:
        """Verify plan completeness against specification."""
        plan_text = "\n".join(
            str(step.description if hasattr(step, 'description') else step)
            for step in plan
        )

        # Check required actions
        for required in spec.required_actions:
            if required.lower() not in plan_text.lower():
                result.add_violation(VerificationViolation(
                    violation_type=ViolationType.INCOMPLETE_PLAN,
                    severity="error",
                    message=f"Required action missing: {required}",
                    suggestion=f"Add a step to {required}",
                ))

        # Check forbidden actions
        for forbidden in spec.forbidden_actions:
            if forbidden.lower() in plan_text.lower():
                result.add_violation(VerificationViolation(
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity="error",
                    message=f"Forbidden action detected: {forbidden}",
                    suggestion=f"Remove or replace steps involving {forbidden}",
                ))

        # Check required outputs via LLM
        if spec.required_outputs:
            prompt = f"""Does this plan produce all required outputs?

Plan:
{plan_text}

Required outputs:
{chr(10).join(f"- {o}" for o in spec.required_outputs)}

For each required output, answer YES or NO and briefly explain.
Format: <output>: YES/NO - <explanation>"""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            # Check for NO responses
            for line in response.choices[0].message.content.split("\n"):
                if ": NO" in line.upper():
                    output = line.split(":")[0].strip()
                    result.add_violation(VerificationViolation(
                        violation_type=ViolationType.INCOMPLETE_PLAN,
                        severity="warning",
                        message=f"Required output may not be produced: {output}",
                        suggestion=f"Add steps to produce {output}",
                    ))

    def _detect_cycles(
        self,
        graph: Dict[str, Set[str]],
    ) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS."""
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def _parse_issues(self, text: str) -> List[VerificationViolation]:
        """Parse issues from LLM response."""
        issues = []
        current_issue = None
        current_severity = "warning"
        current_suggestion = None

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("ISSUE:"):
                if current_issue:
                    issues.append(VerificationViolation(
                        violation_type=ViolationType.SAFETY_VIOLATION,
                        severity=current_severity,
                        message=current_issue,
                        suggestion=current_suggestion,
                    ))
                current_issue = line.split(":", 1)[1].strip()
                current_severity = "warning"
                current_suggestion = None

            elif upper.startswith("SEVERITY:"):
                sev = line.split(":", 1)[1].strip().lower()
                if sev in ("error", "warning", "info"):
                    current_severity = sev

            elif upper.startswith("SUGGESTION:"):
                current_suggestion = line.split(":", 1)[1].strip()

        # Add last issue
        if current_issue:
            issues.append(VerificationViolation(
                violation_type=ViolationType.SAFETY_VIOLATION,
                severity=current_severity,
                message=current_issue,
                suggestion=current_suggestion,
            ))

        return issues

    async def suggest_fixes(
        self,
        plan: List[Any],
        result: VerificationResult,
    ) -> List[str]:
        """Generate suggestions to fix verification failures."""
        if result.is_valid:
            return []

        violations_text = "\n".join(
            f"- {v.message}" for v in result.violations
        )

        plan_text = "\n".join(
            f"- {step.description if hasattr(step, 'description') else step}"
            for step in plan
        )

        prompt = f"""A plan failed verification. Suggest specific fixes.

Plan:
{plan_text}

Violations:
{violations_text}

Provide specific, actionable suggestions to fix each violation.
Number each suggestion."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse suggestions
        suggestions = []
        for line in response.choices[0].message.content.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering/bullet
                if line[0].isdigit():
                    line = line.split(".", 1)[1].strip() if "." in line else line
                else:
                    line = line[1:].strip()
                if line:
                    suggestions.append(line)

        return suggestions
