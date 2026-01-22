"""
AgentCoder - Three-Agent Validation Pattern

Based on research for improved code generation:
- Programmer Agent: Writes code solutions
- Test Designer Agent: Creates comprehensive tests
- Test Executor Agent: Runs tests and validates

This pattern ensures code quality through iterative
feedback between coding and testing.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import litellm
from pydantic import BaseModel


class ValidationStatus(str, Enum):
    """Status of code validation."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class CodeSolution:
    """A code solution from the Programmer Agent."""
    id: str
    code: str
    language: str
    explanation: str
    iteration: int = 0
    test_results: Optional[Dict[str, Any]] = None
    validation_status: ValidationStatus = ValidationStatus.PARTIAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "language": self.language,
            "iteration": self.iteration,
            "status": self.validation_status.value,
            "code_lines": len(self.code.split("\n")),
        }


@dataclass
class TestSuite:
    """A test suite from the Test Designer Agent."""
    id: str
    test_code: str
    test_cases: List[Dict[str, Any]]
    language: str
    target_coverage: float = 0.8
    execution_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "num_tests": len(self.test_cases),
            "language": self.language,
            "target_coverage": self.target_coverage,
        }


class TestResult(BaseModel):
    """Result from test execution."""
    passed: int
    failed: int
    errors: int
    total: int
    coverage: float = 0.0
    failure_details: List[Dict[str, str]] = []
    execution_time: float = 0.0


class ProgrammerAgent:
    """
    Programmer Agent - Generates code solutions.

    Responsibilities:
    - Write code to solve given problems
    - Incorporate feedback from test failures
    - Iterate until tests pass
    - Explain code decisions
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self._solution_counter = 0

    async def generate_solution(
        self,
        task: str,
        context: Dict[str, Any],
        test_feedback: Optional[TestResult] = None,
        previous_solution: Optional[CodeSolution] = None,
    ) -> CodeSolution:
        """
        Generate a code solution.

        Args:
            task: Task description
            context: Context including language, constraints
            test_feedback: Feedback from previous test execution
            previous_solution: Previous solution to improve upon

        Returns:
            CodeSolution with code and explanation
        """
        self._solution_counter += 1
        iteration = (previous_solution.iteration + 1) if previous_solution else 0

        # Build prompt based on iteration
        if test_feedback and previous_solution:
            prompt = self._build_refinement_prompt(
                task, context, test_feedback, previous_solution
            )
        else:
            prompt = self._build_initial_prompt(task, context)

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.choices[0].message.content

        # Extract code and explanation
        code, explanation = self._parse_response(text)
        language = context.get("language", "python")

        return CodeSolution(
            id=f"sol_{self._solution_counter}",
            code=code,
            language=language,
            explanation=explanation,
            iteration=iteration,
        )

    def _build_initial_prompt(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for initial solution."""
        language = context.get("language", "python")
        constraints = context.get("constraints", [])

        prompt = f"""Write a {language} solution for this task.

Task: {task}

Language: {language}
{f"Constraints: {', '.join(constraints)}" if constraints else ""}

Requirements:
1. Write clean, efficient code
2. Handle edge cases appropriately
3. Add brief comments for complex logic
4. Follow {language} best practices

Provide:
1. The complete code solution
2. Brief explanation of your approach

Format:
```{language}
<your code here>
```

EXPLANATION:
<your explanation>"""

        return prompt

    def _build_refinement_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        test_feedback: TestResult,
        previous_solution: CodeSolution,
    ) -> str:
        """Build prompt for refining solution based on test feedback."""
        language = context.get("language", "python")

        # Format failure details
        failures = ""
        if test_feedback.failure_details:
            failures = "\n".join(
                f"- {f['test_name']}: {f['message']}"
                for f in test_feedback.failure_details[:5]
            )

        prompt = f"""Fix your {language} solution based on test results.

Task: {task}

Previous solution:
```{language}
{previous_solution.code}
```

Test Results:
- Passed: {test_feedback.passed}/{test_feedback.total}
- Failed: {test_feedback.failed}
- Errors: {test_feedback.errors}

Failure details:
{failures}

Fix the issues and provide an updated solution:
```{language}
<your fixed code here>
```

EXPLANATION:
<what you fixed and why>"""

        return prompt

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """Parse code and explanation from response."""
        code = ""
        explanation = ""

        # Extract code block
        in_code = False
        code_lines = []
        explanation_lines = []

        for line in text.split("\n"):
            if line.startswith("```"):
                if in_code:
                    in_code = False
                else:
                    in_code = True
            elif in_code:
                code_lines.append(line)
            elif "EXPLANATION:" in line.upper():
                continue
            elif not in_code and code_lines:  # After code block
                explanation_lines.append(line)

        code = "\n".join(code_lines)
        explanation = "\n".join(explanation_lines).strip()

        return code, explanation


class TestDesignerAgent:
    """
    Test Designer Agent - Creates comprehensive tests.

    Responsibilities:
    - Analyze code to identify test cases
    - Design tests for edge cases
    - Ensure good coverage
    - Create both positive and negative tests
    """

    def __init__(
        self,
        model: str = "gpt-4",
    ):
        self.model = model
        self._suite_counter = 0

    async def design_tests(
        self,
        task: str,
        solution: CodeSolution,
        context: Dict[str, Any],
    ) -> TestSuite:
        """
        Design tests for a code solution.

        Args:
            task: Original task description
            solution: Code solution to test
            context: Context including language, constraints

        Returns:
            TestSuite with test code and cases
        """
        self._suite_counter += 1

        prompt = self._build_prompt(task, solution, context)

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.choices[0].message.content

        # Parse tests
        test_code, test_cases = self._parse_tests(text, solution.language)

        return TestSuite(
            id=f"suite_{self._suite_counter}",
            test_code=test_code,
            test_cases=test_cases,
            language=solution.language,
        )

    def _build_prompt(
        self,
        task: str,
        solution: CodeSolution,
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for test design."""
        language = solution.language
        framework = self._get_test_framework(language)

        prompt = f"""Design comprehensive tests for this code.

Task: {task}

Code to test:
```{language}
{solution.code}
```

Test framework: {framework}

Create tests that cover:
1. Normal cases (happy path)
2. Edge cases (empty input, large input, etc.)
3. Error cases (invalid input, error conditions)
4. Boundary conditions

For each test, provide:
- Test name
- What it tests
- Expected outcome

Then provide the complete test code.

TEST CASES:
<list each test case>

TEST CODE:
```{language}
<complete test code>
```"""

        return prompt

    def _parse_tests(
        self,
        text: str,
        language: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse test code and cases from response."""
        test_code = ""
        test_cases = []

        # Extract test cases
        in_cases = False
        in_code = False
        code_lines = []

        for line in text.split("\n"):
            if "TEST CASES:" in line.upper():
                in_cases = True
                continue
            elif "TEST CODE:" in line.upper():
                in_cases = False
                continue
            elif line.startswith("```"):
                if in_code:
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_lines.append(line)
            elif in_cases and line.strip():
                # Parse test case
                if line.strip().startswith(("-", "*", "•")):
                    test_cases.append({
                        "description": line.strip().lstrip("-*• "),
                        "type": "unit",
                    })
                elif line.strip()[0].isdigit():
                    test_cases.append({
                        "description": line.strip().lstrip("0123456789.) "),
                        "type": "unit",
                    })

        test_code = "\n".join(code_lines)

        return test_code, test_cases

    def _get_test_framework(self, language: str) -> str:
        """Get appropriate test framework for language."""
        frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "java": "JUnit",
            "go": "testing",
            "rust": "built-in test",
            "cpp": "GoogleTest",
            "c": "Unity",
        }
        return frameworks.get(language.lower(), "unit test")


class TestExecutorAgent:
    """
    Test Executor Agent - Runs tests and reports results.

    Responsibilities:
    - Execute test suites
    - Collect and format results
    - Report coverage metrics
    - Provide actionable feedback
    """

    def __init__(
        self,
        model: str = "gpt-4",
        sandbox_executor: Optional[Any] = None,
    ):
        self.model = model
        self.sandbox_executor = sandbox_executor

    async def execute_tests(
        self,
        solution: CodeSolution,
        test_suite: TestSuite,
        context: Dict[str, Any],
    ) -> TestResult:
        """
        Execute tests against solution.

        Args:
            solution: Code solution to test
            test_suite: Test suite to run
            context: Execution context

        Returns:
            TestResult with pass/fail details
        """
        # If we have a sandbox executor, use it
        if self.sandbox_executor:
            return await self._execute_in_sandbox(solution, test_suite, context)

        # Otherwise, use LLM to simulate execution
        return await self._simulate_execution(solution, test_suite, context)

    async def _execute_in_sandbox(
        self,
        solution: CodeSolution,
        test_suite: TestSuite,
        context: Dict[str, Any],
    ) -> TestResult:
        """Execute tests in sandbox environment."""
        # This would use the actual sandbox
        # For now, delegate to simulation
        return await self._simulate_execution(solution, test_suite, context)

    async def _simulate_execution(
        self,
        solution: CodeSolution,
        test_suite: TestSuite,
        context: Dict[str, Any],
    ) -> TestResult:
        """Simulate test execution using LLM."""
        prompt = f"""Analyze what would happen if we ran these tests against this code.

Code:
```{solution.language}
{solution.code}
```

Tests:
```{test_suite.language}
{test_suite.test_code}
```

For each test, determine:
1. Would it pass or fail?
2. If fail, what's the error message?

Provide results in this format:
PASSED: <number>
FAILED: <number>
ERRORS: <number>
TOTAL: {len(test_suite.test_cases)}
COVERAGE: <estimated percentage>

FAILURES:
- <test_name>: <failure reason>
...

Analysis:"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        # Parse results
        return self._parse_results(
            response.choices[0].message.content,
            len(test_suite.test_cases)
        )

    def _parse_results(self, text: str, total_tests: int) -> TestResult:
        """Parse test results from response."""
        passed = 0
        failed = 0
        errors = 0
        coverage = 0.0
        failures = []

        lines = text.strip().split("\n")
        in_failures = False

        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("PASSED:"):
                try:
                    passed = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif upper.startswith("FAILED:"):
                try:
                    failed = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif upper.startswith("ERRORS:"):
                try:
                    errors = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif upper.startswith("COVERAGE:"):
                try:
                    cov_str = line.split(":")[1].strip().rstrip("%")
                    coverage = float(cov_str) / 100 if float(cov_str) > 1 else float(cov_str)
                except (ValueError, IndexError):
                    pass
            elif "FAILURES:" in upper:
                in_failures = True
            elif in_failures and line.startswith("-"):
                parts = line[1:].strip().split(":", 1)
                if len(parts) == 2:
                    failures.append({
                        "test_name": parts[0].strip(),
                        "message": parts[1].strip(),
                    })

        return TestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            total=total_tests,
            coverage=coverage,
            failure_details=failures,
        )


class AgentCoder:
    """
    AgentCoder - Three-agent code validation system.

    Coordinates:
    - Programmer Agent: Writes code
    - Test Designer Agent: Creates tests
    - Test Executor Agent: Runs tests

    Iterates until tests pass or max iterations reached.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
        target_coverage: float = 0.8,
        sandbox_executor: Optional[Any] = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage

        self.programmer = ProgrammerAgent(model, max_iterations)
        self.test_designer = TestDesignerAgent(model)
        self.test_executor = TestExecutorAgent(model, sandbox_executor)

        self.history: List[Dict[str, Any]] = []

    async def solve(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Solve a coding task with test validation.

        Args:
            task: Task description
            context: Optional context (language, constraints, etc.)

        Returns:
            Dictionary with solution, tests, and results
        """
        context = context or {"language": "python"}
        solution = None
        test_suite = None
        test_result = None

        for iteration in range(self.max_iterations):
            # Generate or refine solution
            solution = await self.programmer.generate_solution(
                task=task,
                context=context,
                test_feedback=test_result,
                previous_solution=solution,
            )

            # Design tests (only on first iteration or if solution changed significantly)
            if test_suite is None or iteration == 0:
                test_suite = await self.test_designer.design_tests(
                    task=task,
                    solution=solution,
                    context=context,
                )

            # Execute tests
            test_result = await self.test_executor.execute_tests(
                solution=solution,
                test_suite=test_suite,
                context=context,
            )

            # Record iteration
            self.history.append({
                "iteration": iteration,
                "solution_id": solution.id,
                "test_suite_id": test_suite.id,
                "passed": test_result.passed,
                "failed": test_result.failed,
                "coverage": test_result.coverage,
            })

            # Check if we're done
            if test_result.failed == 0 and test_result.errors == 0:
                solution.validation_status = ValidationStatus.PASSED
                break

            # Check coverage target
            if (test_result.failed == 0 and
                test_result.coverage >= self.target_coverage):
                solution.validation_status = ValidationStatus.PASSED
                break

        # Final status
        if solution.validation_status != ValidationStatus.PASSED:
            if test_result.passed > 0:
                solution.validation_status = ValidationStatus.PARTIAL
            else:
                solution.validation_status = ValidationStatus.FAILED

        solution.test_results = {
            "passed": test_result.passed,
            "failed": test_result.failed,
            "errors": test_result.errors,
            "coverage": test_result.coverage,
        }

        return {
            "success": solution.validation_status == ValidationStatus.PASSED,
            "solution": {
                "code": solution.code,
                "language": solution.language,
                "explanation": solution.explanation,
            },
            "tests": {
                "code": test_suite.test_code,
                "cases": test_suite.test_cases,
            },
            "results": {
                "passed": test_result.passed,
                "failed": test_result.failed,
                "errors": test_result.errors,
                "total": test_result.total,
                "coverage": test_result.coverage,
            },
            "iterations": len(self.history),
            "status": solution.validation_status.value,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get iteration history."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear iteration history."""
        self.history.clear()
