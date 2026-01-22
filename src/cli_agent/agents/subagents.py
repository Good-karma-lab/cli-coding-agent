"""
Specialized SubAgents - Planner, Coder, Tester, Reviewer, Researcher

Based on research for specialized agent roles:
- Each agent has focused expertise
- Agents communicate via structured messages
- Supports both autonomous and orchestrated modes
- Integrates with memory and code understanding systems
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from enum import Enum
from datetime import datetime
import litellm
from pydantic import BaseModel


class AgentRole(str, Enum):
    """Roles for specialized agents."""
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    DEBUGGER = "debugger"
    REFACTORER = "refactorer"


@dataclass
class AgentContext:
    """Context passed to agent for task execution."""
    task: str
    codebase_info: Dict[str, Any] = field(default_factory=dict)
    relevant_files: List[str] = field(default_factory=list)
    previous_results: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    memory_context: Optional[str] = None


class AgentOutput(BaseModel):
    """Structured output from an agent."""
    success: bool
    content: Any
    reasoning: str = ""
    artifacts: Dict[str, Any] = {}  # Files created, tests written, etc.
    next_steps: List[str] = []
    confidence: float = 0.8


class BaseSubAgent(ABC):
    """
    Base class for specialized sub-agents.

    Provides common functionality for all agents:
    - LLM interaction
    - Context management
    - Output formatting
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.5,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.history: List[Dict[str, str]] = []

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Agent's role."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Agent's system prompt."""
        pass

    @property
    def capabilities(self) -> List[str]:
        """Agent's capabilities."""
        return []

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Execute a task.

        Args:
            task: Task description
            context: Optional context dictionary

        Returns:
            AgentOutput with results
        """
        context = context or {}
        agent_context = AgentContext(
            task=task,
            codebase_info=context.get("codebase_info", {}),
            relevant_files=context.get("relevant_files", []),
            previous_results=context.get("previous_results", {}),
            constraints=context.get("constraints", []),
            memory_context=context.get("memory_context"),
        )

        for attempt in range(self.max_retries):
            try:
                result = await self._execute_impl(agent_context)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return AgentOutput(
                        success=False,
                        content=None,
                        reasoning=f"Failed after {self.max_retries} attempts: {str(e)}",
                        confidence=0.0,
                    )
                await asyncio.sleep(1)  # Brief delay before retry

        return AgentOutput(success=False, content=None, reasoning="Unknown error")

    @abstractmethod
    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Implementation of task execution."""
        pass

    async def _call_llm(
        self,
        prompt: str,
        system_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
        """Call LLM with prompt."""
        messages = [
            {"role": "system", "content": system_override or self.system_prompt},
            *self.history[-5:],  # Recent history
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature_override or self.temperature,
        )

        result = response.choices[0].message.content.strip()

        # Update history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": result})

        return result

    def _format_context(self, context: AgentContext) -> str:
        """Format context for prompts."""
        parts = [f"Task: {context.task}"]

        if context.relevant_files:
            parts.append(f"Relevant files: {', '.join(context.relevant_files[:10])}")

        if context.codebase_info:
            parts.append(f"Codebase: {context.codebase_info.get('name', 'unknown')}")

        if context.constraints:
            parts.append(f"Constraints: {', '.join(context.constraints)}")

        if context.memory_context:
            parts.append(f"Memory context: {context.memory_context[:200]}")

        if context.previous_results:
            parts.append("Previous results available")

        return "\n".join(parts)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()


class PlannerAgent(BaseSubAgent):
    """
    Planner Agent - Creates execution plans for tasks.

    Specializes in:
    - Task decomposition
    - Dependency analysis
    - Resource estimation
    - Risk identification
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.PLANNER

    @property
    def system_prompt(self) -> str:
        return """You are a Planning Agent specialized in creating execution plans for software tasks.

Your responsibilities:
1. Analyze tasks and break them into clear, actionable steps
2. Identify dependencies between steps
3. Estimate complexity and potential risks
4. Consider edge cases and failure modes
5. Create plans that are specific enough to execute

When creating plans:
- Be specific and concrete
- Consider the existing codebase
- Identify which files need modification
- Note potential blockers or risks
- Order steps logically with dependencies"""

    @property
    def capabilities(self) -> List[str]:
        return ["planning", "analysis", "decomposition"]

    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Create an execution plan."""
        prompt = f"""Create an execution plan for this task.

{self._format_context(context)}

Provide:
1. A list of ordered steps (numbered)
2. Dependencies between steps
3. Estimated complexity (low/medium/high) for each step
4. Potential risks or blockers
5. Files that may need modification

Format:
STEPS:
1. <step description> [Complexity: low/medium/high]
2. <step description> [Complexity: low/medium/high]
...

DEPENDENCIES:
- Step X depends on Step Y
...

RISKS:
- <risk description>
...

FILES:
- <file path>
...

Plan:"""

        response = await self._call_llm(prompt)

        # Parse plan
        plan = self._parse_plan(response)

        return AgentOutput(
            success=True,
            content=plan,
            reasoning="Plan created based on task analysis",
            artifacts={"plan_text": response},
            next_steps=plan.get("steps", [])[:3],
            confidence=0.85,
        )

    def _parse_plan(self, text: str) -> Dict[str, Any]:
        """Parse plan from response."""
        plan = {
            "steps": [],
            "dependencies": [],
            "risks": [],
            "files": [],
        }

        current_section = None
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            upper = line.upper()

            if "STEPS:" in upper:
                current_section = "steps"
            elif "DEPENDENCIES:" in upper:
                current_section = "dependencies"
            elif "RISKS:" in upper:
                current_section = "risks"
            elif "FILES:" in upper:
                current_section = "files"
            elif line and current_section:
                # Remove numbering and bullets
                clean = line.lstrip("0123456789.-) ")
                if clean:
                    plan[current_section].append(clean)

        return plan


class CoderAgent(BaseSubAgent):
    """
    Coder Agent - Writes and modifies code.

    Specializes in:
    - Code generation
    - Code modification
    - Following coding standards
    - Implementing designs
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.CODER

    @property
    def system_prompt(self) -> str:
        return """You are a Coding Agent specialized in writing high-quality code.

Your responsibilities:
1. Write clean, well-structured code
2. Follow established patterns in the codebase
3. Add appropriate comments and documentation
4. Handle errors gracefully
5. Consider edge cases

Coding standards:
- Use clear, descriptive names
- Keep functions focused and small
- Write self-documenting code
- Follow language idioms and best practices
- Ensure type safety where applicable"""

    @property
    def capabilities(self) -> List[str]:
        return ["coding", "implementation", "refactoring"]

    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Write or modify code."""
        prompt = f"""Write code for this task.

{self._format_context(context)}

Provide:
1. The code implementation
2. Brief explanation of key decisions
3. Any assumptions made
4. Suggestions for testing

Code:"""

        response = await self._call_llm(prompt, temperature_override=0.3)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)

        return AgentOutput(
            success=True,
            content={
                "code": code_blocks,
                "explanation": response,
            },
            reasoning="Code written based on task requirements",
            artifacts={"code_blocks": code_blocks},
            next_steps=["Review code", "Write tests", "Integrate changes"],
            confidence=0.8,
        )

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from response."""
        blocks = []
        in_block = False
        current_block = {"language": "", "code": ""}

        lines = text.split("\n")
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    # End of block
                    if current_block["code"].strip():
                        blocks.append(current_block)
                    current_block = {"language": "", "code": ""}
                    in_block = False
                else:
                    # Start of block
                    in_block = True
                    current_block["language"] = line[3:].strip()
            elif in_block:
                current_block["code"] += line + "\n"

        return blocks


class TesterAgent(BaseSubAgent):
    """
    Tester Agent - Creates and runs tests.

    Specializes in:
    - Test case design
    - Test implementation
    - Coverage analysis
    - Edge case identification
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.TESTER

    @property
    def system_prompt(self) -> str:
        return """You are a Testing Agent specialized in software testing.

Your responsibilities:
1. Design comprehensive test cases
2. Write clear, maintainable tests
3. Identify edge cases and boundary conditions
4. Ensure good test coverage
5. Create both unit and integration tests

Testing principles:
- Test behavior, not implementation
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Include positive and negative cases
- Test error handling paths"""

    @property
    def capabilities(self) -> List[str]:
        return ["testing", "quality assurance", "validation"]

    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Create tests for code."""
        prompt = f"""Create tests for this task.

{self._format_context(context)}

Provide:
1. Test cases with descriptions
2. Test code implementation
3. Expected coverage areas
4. Edge cases covered

Tests:"""

        response = await self._call_llm(prompt)

        # Extract test information
        test_cases = self._parse_test_cases(response)
        code_blocks = self._extract_code_blocks(response)

        return AgentOutput(
            success=True,
            content={
                "test_cases": test_cases,
                "test_code": code_blocks,
            },
            reasoning="Tests designed based on functionality requirements",
            artifacts={
                "test_cases": test_cases,
                "code_blocks": code_blocks,
            },
            next_steps=["Run tests", "Check coverage", "Fix failing tests"],
            confidence=0.85,
        )

    def _parse_test_cases(self, text: str) -> List[Dict[str, str]]:
        """Parse test cases from response."""
        cases = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            # Look for test descriptions
            if line.lower().startswith("test"):
                cases.append({
                    "description": line,
                    "type": "unit" if "unit" in line.lower() else "integration",
                })
            elif "should" in line.lower():
                cases.append({
                    "description": line,
                    "type": "unit",
                })

        return cases

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from response."""
        blocks = []
        in_block = False
        current_block = {"language": "", "code": ""}

        lines = text.split("\n")
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    if current_block["code"].strip():
                        blocks.append(current_block)
                    current_block = {"language": "", "code": ""}
                    in_block = False
                else:
                    in_block = True
                    current_block["language"] = line[3:].strip()
            elif in_block:
                current_block["code"] += line + "\n"

        return blocks


class ReviewerAgent(BaseSubAgent):
    """
    Reviewer Agent - Reviews code for quality and issues.

    Specializes in:
    - Code review
    - Security analysis
    - Performance review
    - Best practice verification
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.REVIEWER

    @property
    def system_prompt(self) -> str:
        return """You are a Code Review Agent specialized in reviewing software quality.

Your responsibilities:
1. Review code for correctness and clarity
2. Identify potential bugs and issues
3. Check for security vulnerabilities
4. Assess performance implications
5. Verify adherence to best practices

Review criteria:
- Correctness: Does it do what it's supposed to?
- Security: Are there vulnerabilities?
- Performance: Are there efficiency issues?
- Maintainability: Is it easy to understand and modify?
- Style: Does it follow conventions?"""

    @property
    def capabilities(self) -> List[str]:
        return ["reviewing", "quality analysis", "security"]

    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Review code or changes."""
        prompt = f"""Review this code/changes.

{self._format_context(context)}

Provide:
1. Overall assessment (approve/request changes/reject)
2. Issues found (categorized by severity)
3. Suggestions for improvement
4. Security concerns if any
5. Performance considerations

Review:"""

        response = await self._call_llm(prompt)

        # Parse review
        review = self._parse_review(response)

        # Determine approval status
        approval = "approve"
        if review.get("critical_issues"):
            approval = "reject"
        elif review.get("major_issues"):
            approval = "request_changes"

        return AgentOutput(
            success=True,
            content={
                "approval": approval,
                "review": review,
            },
            reasoning="Code reviewed against quality criteria",
            artifacts={"review_text": response},
            next_steps=review.get("suggestions", [])[:3],
            confidence=0.9,
        )

    def _parse_review(self, text: str) -> Dict[str, Any]:
        """Parse review from response."""
        review = {
            "critical_issues": [],
            "major_issues": [],
            "minor_issues": [],
            "suggestions": [],
            "security": [],
            "performance": [],
        }

        current_section = None
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            lower = line.lower()

            if "critical" in lower:
                current_section = "critical_issues"
            elif "major" in lower:
                current_section = "major_issues"
            elif "minor" in lower:
                current_section = "minor_issues"
            elif "suggest" in lower:
                current_section = "suggestions"
            elif "security" in lower:
                current_section = "security"
            elif "performance" in lower:
                current_section = "performance"
            elif line.startswith("-") and current_section:
                review[current_section].append(line[1:].strip())

        return review


class ResearcherAgent(BaseSubAgent):
    """
    Researcher Agent - Researches codebases and documentation.

    Specializes in:
    - Codebase exploration
    - Documentation lookup
    - Pattern identification
    - Dependency analysis
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.RESEARCHER

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent specialized in exploring and understanding codebases.

Your responsibilities:
1. Explore codebase structure and patterns
2. Find relevant code and documentation
3. Analyze dependencies and relationships
4. Identify conventions and standards
5. Summarize findings clearly

Research approach:
- Start with high-level structure
- Follow call chains and imports
- Note patterns and conventions
- Document key findings
- Provide actionable insights"""

    @property
    def capabilities(self) -> List[str]:
        return ["researching", "analysis", "documentation"]

    async def _execute_impl(self, context: AgentContext) -> AgentOutput:
        """Research the codebase."""
        prompt = f"""Research and analyze for this task.

{self._format_context(context)}

Provide:
1. Relevant files and their purposes
2. Key patterns or conventions found
3. Dependencies and relationships
4. Potential areas of impact
5. Recommendations for approach

Findings:"""

        response = await self._call_llm(prompt)

        # Parse findings
        findings = self._parse_findings(response)

        return AgentOutput(
            success=True,
            content=findings,
            reasoning="Research conducted based on task requirements",
            artifacts={"findings_text": response},
            next_steps=findings.get("recommendations", [])[:3],
            confidence=0.85,
        )

    def _parse_findings(self, text: str) -> Dict[str, Any]:
        """Parse findings from response."""
        findings = {
            "files": [],
            "patterns": [],
            "dependencies": [],
            "impact_areas": [],
            "recommendations": [],
        }

        current_section = None
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            lower = line.lower()

            if "file" in lower and ":" in line:
                current_section = "files"
            elif "pattern" in lower:
                current_section = "patterns"
            elif "depend" in lower:
                current_section = "dependencies"
            elif "impact" in lower:
                current_section = "impact_areas"
            elif "recommend" in lower:
                current_section = "recommendations"
            elif line.startswith("-") and current_section:
                findings[current_section].append(line[1:].strip())

        return findings
