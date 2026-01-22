"""
Multi-Agent Reflexion (MAR) - Diverse Reasoning Personas

Based on research for improved agent reasoning:
- Multiple agents with distinct personas analyze problems
- Acting agent: Executes actions
- Diagnosing agent: Identifies issues
- Critiquing agent: Evaluates solutions
- Aggregating agent: Synthesizes insights
- Memory sharing for collective learning
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
import litellm
from pydantic import BaseModel


class PersonaType(str, Enum):
    """Types of reflexion personas."""
    ACTOR = "actor"           # Takes actions and executes
    DIAGNOSER = "diagnoser"   # Identifies problems and root causes
    CRITIC = "critic"         # Evaluates and critiques solutions
    AGGREGATOR = "aggregator" # Synthesizes multiple perspectives
    OPTIMIST = "optimist"     # Focuses on possibilities and solutions
    PESSIMIST = "pessimist"   # Identifies risks and failure modes
    EXPERT = "expert"         # Domain-specific expertise
    NOVICE = "novice"         # Fresh perspective, asks basic questions


@dataclass
class ReflexionPersona:
    """
    A persona for multi-agent reflexion.

    Each persona has a unique perspective and reasoning style
    that contributes to collective problem solving.
    """
    persona_type: PersonaType
    name: str
    description: str
    system_prompt: str

    # Persona-specific parameters
    temperature: float = 0.7
    focus_areas: List[str] = field(default_factory=list)

    # Conversation history for this persona
    history: List[Dict[str, str]] = field(default_factory=list)

    # Accumulated insights
    insights: List[str] = field(default_factory=list)

    def add_to_history(self, role: str, content: str) -> None:
        """Add message to persona's history."""
        self.history.append({"role": role, "content": content})

    def add_insight(self, insight: str) -> None:
        """Add an insight from this persona."""
        self.insights.append(insight)

    def get_recent_history(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.history[-n:] if len(self.history) > n else self.history

    def reset(self) -> None:
        """Reset persona state."""
        self.history.clear()
        self.insights.clear()


class ReflexionResult(BaseModel):
    """Result of a reflexion round."""
    task: str
    round_number: int
    persona_outputs: Dict[str, str]
    aggregated_insight: str
    action_recommendation: str
    confidence: float
    dissenting_views: List[str] = []
    open_questions: List[str] = []


class MultiAgentReflexion:
    """
    Multi-Agent Reflexion (MAR) system.

    Implements collaborative reasoning with diverse personas:
    - Each persona analyzes the problem from their perspective
    - Insights are shared and aggregated
    - Collective wisdom guides decision making
    - Memory enables learning from past attempts
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
        enable_memory_sharing: bool = True,
    ):
        self.model = model
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.enable_memory_sharing = enable_memory_sharing

        self.personas: Dict[PersonaType, ReflexionPersona] = {}
        self.shared_memory: List[Dict[str, Any]] = []
        self.reflexion_history: List[ReflexionResult] = []

        # Initialize default personas
        self._initialize_default_personas()

    def _initialize_default_personas(self) -> None:
        """Initialize the default set of personas."""
        persona_configs = [
            {
                "type": PersonaType.ACTOR,
                "name": "ActionAgent",
                "description": "Executes actions and implements solutions",
                "system_prompt": """You are the Action Agent. Your role is to:
- Execute concrete actions to solve problems
- Implement solutions based on collective insights
- Report results and observations accurately
- Be pragmatic and focus on what can be done now

When analyzing a task, propose specific actions with clear steps.""",
                "focus_areas": ["implementation", "execution", "results"],
            },
            {
                "type": PersonaType.DIAGNOSER,
                "name": "DiagnosticAgent",
                "description": "Identifies problems and root causes",
                "system_prompt": """You are the Diagnostic Agent. Your role is to:
- Identify the root cause of problems
- Analyze symptoms and trace them to their source
- Ask probing questions to understand issues deeply
- Connect related problems to find patterns

When analyzing a task, focus on what's wrong and why.""",
                "focus_areas": ["root cause", "patterns", "dependencies"],
            },
            {
                "type": PersonaType.CRITIC,
                "name": "CriticAgent",
                "description": "Evaluates and critiques solutions",
                "system_prompt": """You are the Critic Agent. Your role is to:
- Evaluate proposed solutions critically but constructively
- Identify weaknesses, edge cases, and potential failures
- Ensure solutions are robust and well-thought-out
- Challenge assumptions and identify blind spots

When analyzing a task, focus on what could go wrong and how to prevent it.""",
                "focus_areas": ["weaknesses", "edge cases", "robustness"],
            },
            {
                "type": PersonaType.AGGREGATOR,
                "name": "SynthesisAgent",
                "description": "Synthesizes insights from all perspectives",
                "system_prompt": """You are the Synthesis Agent. Your role is to:
- Aggregate insights from all other agents
- Find common ground and resolve conflicts
- Create a coherent recommendation from diverse inputs
- Identify the most valuable insights and actionable items

When analyzing perspectives, create a unified, balanced view.""",
                "focus_areas": ["synthesis", "consensus", "recommendations"],
            },
        ]

        for config in persona_configs:
            persona = ReflexionPersona(
                persona_type=config["type"],
                name=config["name"],
                description=config["description"],
                system_prompt=config["system_prompt"],
                focus_areas=config["focus_areas"],
            )
            self.personas[config["type"]] = persona

    def add_persona(self, persona: ReflexionPersona) -> None:
        """Add a custom persona."""
        self.personas[persona.persona_type] = persona

    async def reflect(
        self,
        task: str,
        context: Dict[str, Any],
        previous_attempt: Optional[Dict[str, Any]] = None,
    ) -> ReflexionResult:
        """
        Run a reflexion cycle on a task.

        Args:
            task: Task description
            context: Current context/state
            previous_attempt: Results from previous attempt if any

        Returns:
            ReflexionResult with aggregated insights
        """
        round_number = len(self.reflexion_history) + 1
        persona_outputs: Dict[str, str] = {}

        # Prepare context with shared memory
        enhanced_context = self._enhance_context(context, previous_attempt)

        # Phase 1: Individual Analysis
        # Run personas in parallel
        analysis_tasks = [
            self._run_persona_analysis(persona, task, enhanced_context)
            for persona in self.personas.values()
            if persona.persona_type != PersonaType.AGGREGATOR
        ]

        results = await asyncio.gather(*analysis_tasks)

        for persona_type, output in results:
            persona_outputs[persona_type.value] = output

        # Phase 2: Cross-Persona Discussion (optional for deeper reflection)
        if self.max_rounds > 1:
            discussion_outputs = await self._run_discussion(
                task, enhanced_context, persona_outputs
            )
            for persona_type, output in discussion_outputs.items():
                persona_outputs[persona_type] = output

        # Phase 3: Aggregation
        aggregator = self.personas.get(PersonaType.AGGREGATOR)
        if aggregator:
            aggregated = await self._aggregate_insights(
                aggregator, task, enhanced_context, persona_outputs
            )
        else:
            aggregated = self._simple_aggregate(persona_outputs)

        # Phase 4: Generate Action Recommendation
        action_rec, confidence = await self._generate_recommendation(
            task, enhanced_context, aggregated, persona_outputs
        )

        # Identify dissenting views and open questions
        dissenting = self._identify_dissent(persona_outputs)
        questions = self._identify_questions(persona_outputs)

        # Create result
        result = ReflexionResult(
            task=task,
            round_number=round_number,
            persona_outputs=persona_outputs,
            aggregated_insight=aggregated,
            action_recommendation=action_rec,
            confidence=confidence,
            dissenting_views=dissenting,
            open_questions=questions,
        )

        # Store in history and shared memory
        self.reflexion_history.append(result)
        if self.enable_memory_sharing:
            self._update_shared_memory(result)

        return result

    async def _run_persona_analysis(
        self,
        persona: ReflexionPersona,
        task: str,
        context: Dict[str, Any],
    ) -> tuple[PersonaType, str]:
        """Run analysis for a single persona."""
        # Build prompt with persona's perspective
        prompt = f"""Analyze this task from your perspective.

Task: {task}

Context:
{self._format_context(context)}

Previous insights from this session:
{self._format_insights(persona.insights)}

Provide your analysis focusing on: {', '.join(persona.focus_areas)}

Your analysis:"""

        messages = [
            {"role": "system", "content": persona.system_prompt},
            *persona.get_recent_history(),
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=persona.temperature,
        )

        output = response.choices[0].message.content.strip()

        # Update persona history
        persona.add_to_history("user", prompt)
        persona.add_to_history("assistant", output)

        return persona.persona_type, output

    async def _run_discussion(
        self,
        task: str,
        context: Dict[str, Any],
        initial_outputs: Dict[str, str],
    ) -> Dict[str, str]:
        """Run cross-persona discussion for deeper reflection."""
        updated_outputs = {}

        # Each non-aggregator persona responds to others' analyses
        for persona in self.personas.values():
            if persona.persona_type == PersonaType.AGGREGATOR:
                continue

            other_analyses = {
                k: v for k, v in initial_outputs.items()
                if k != persona.persona_type.value
            }

            prompt = f"""Review other agents' analyses and refine your perspective.

Task: {task}

Other analyses:
{self._format_analyses(other_analyses)}

Your original analysis:
{initial_outputs.get(persona.persona_type.value, '')}

Based on the other perspectives, provide your refined analysis.
Note any points of agreement or disagreement.

Refined analysis:"""

            messages = [
                {"role": "system", "content": persona.system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=persona.temperature,
            )

            output = response.choices[0].message.content.strip()
            updated_outputs[persona.persona_type.value] = output

            # Extract and store insights
            insight = self._extract_key_insight(output)
            if insight:
                persona.add_insight(insight)

        return updated_outputs

    async def _aggregate_insights(
        self,
        aggregator: ReflexionPersona,
        task: str,
        context: Dict[str, Any],
        persona_outputs: Dict[str, str],
    ) -> str:
        """Aggregate insights from all personas."""
        prompt = f"""Synthesize the analyses from all agents into a coherent understanding.

Task: {task}

Agent Analyses:
{self._format_analyses(persona_outputs)}

Create a synthesis that:
1. Identifies key points of consensus
2. Highlights valuable unique insights
3. Notes unresolved disagreements
4. Provides a balanced overall assessment

Synthesized insight:"""

        messages = [
            {"role": "system", "content": aggregator.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=0.5,  # Lower temperature for aggregation
        )

        return response.choices[0].message.content.strip()

    def _simple_aggregate(self, persona_outputs: Dict[str, str]) -> str:
        """Simple aggregation when no aggregator persona exists."""
        lines = ["Summary of agent analyses:"]
        for persona_type, output in persona_outputs.items():
            # Extract first sentence or first 100 chars as summary
            summary = output.split('.')[0] if '.' in output else output[:100]
            lines.append(f"- {persona_type}: {summary}")
        return "\n".join(lines)

    async def _generate_recommendation(
        self,
        task: str,
        context: Dict[str, Any],
        aggregated: str,
        persona_outputs: Dict[str, str],
    ) -> tuple[str, float]:
        """Generate action recommendation with confidence score."""
        prompt = f"""Based on the collective analysis, provide an action recommendation.

Task: {task}

Aggregated Analysis:
{aggregated}

Provide:
1. A specific, actionable recommendation
2. A confidence score from 0.0 to 1.0

Format:
RECOMMENDATION: <your recommendation>
CONFIDENCE: <score>"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()

        # Parse recommendation and confidence
        recommendation = ""
        confidence = 0.5

        for line in text.split("\n"):
            if line.upper().startswith("RECOMMENDATION:"):
                recommendation = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    confidence = float(conf_str.split()[0])
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

        if not recommendation:
            recommendation = text  # Use full response if parsing fails

        return recommendation, confidence

    def _identify_dissent(self, persona_outputs: Dict[str, str]) -> List[str]:
        """Identify dissenting views from persona outputs."""
        dissent_markers = [
            "disagree", "however", "but", "concern", "risk",
            "problem with", "issue with", "not sure", "caution"
        ]

        dissenting = []
        for persona_type, output in persona_outputs.items():
            for marker in dissent_markers:
                if marker in output.lower():
                    # Extract the sentence containing the marker
                    sentences = output.split('.')
                    for sentence in sentences:
                        if marker in sentence.lower():
                            dissenting.append(f"{persona_type}: {sentence.strip()}")
                            break
                    break

        return dissenting[:5]  # Limit to top 5

    def _identify_questions(self, persona_outputs: Dict[str, str]) -> List[str]:
        """Identify open questions from persona outputs."""
        questions = []

        for persona_type, output in persona_outputs.items():
            # Find sentences ending with ?
            sentences = output.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence.endswith('?'):
                    questions.append(f"{persona_type}: {sentence}")

        return questions[:5]  # Limit to top 5

    def _extract_key_insight(self, text: str) -> Optional[str]:
        """Extract the key insight from a response."""
        # Look for explicit insight markers
        markers = ["key insight:", "main point:", "importantly:", "notably:"]
        text_lower = text.lower()

        for marker in markers:
            if marker in text_lower:
                idx = text_lower.index(marker)
                insight = text[idx + len(marker):].split('.')[0].strip()
                if insight:
                    return insight

        # Fall back to first sentence
        first_sentence = text.split('.')[0].strip()
        return first_sentence if len(first_sentence) > 20 else None

    def _enhance_context(
        self,
        context: Dict[str, Any],
        previous_attempt: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Enhance context with shared memory and previous attempts."""
        enhanced = context.copy()

        if previous_attempt:
            enhanced["previous_attempt"] = previous_attempt

        if self.shared_memory:
            enhanced["shared_memory_highlights"] = [
                m["key_insight"] for m in self.shared_memory[-3:]
            ]

        return enhanced

    def _update_shared_memory(self, result: ReflexionResult) -> None:
        """Update shared memory with result."""
        self.shared_memory.append({
            "task": result.task,
            "round": result.round_number,
            "key_insight": result.aggregated_insight[:200],
            "recommendation": result.action_recommendation[:200],
            "confidence": result.confidence,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep memory bounded
        if len(self.shared_memory) > 20:
            self.shared_memory = self.shared_memory[-20:]

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts."""
        lines = []
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 200:
                lines.append(f"  {key}: {value[:200]}...")
            elif isinstance(value, (list, dict)):
                lines.append(f"  {key}: {type(value).__name__} ({len(value)} items)")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _format_insights(self, insights: List[str]) -> str:
        """Format insights for prompts."""
        if not insights:
            return "  (no previous insights)"
        return "\n".join(f"  - {insight}" for insight in insights[-5:])

    def _format_analyses(self, analyses: Dict[str, str]) -> str:
        """Format analyses for prompts."""
        lines = []
        for persona_type, analysis in analyses.items():
            lines.append(f"[{persona_type}]:")
            # Truncate long analyses
            if len(analysis) > 500:
                lines.append(f"  {analysis[:500]}...")
            else:
                lines.append(f"  {analysis}")
            lines.append("")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all personas and shared memory."""
        for persona in self.personas.values():
            persona.reset()
        self.shared_memory.clear()
        self.reflexion_history.clear()

    def get_reflexion_summary(self) -> Dict[str, Any]:
        """Get summary of reflexion history."""
        if not self.reflexion_history:
            return {"rounds": 0}

        confidences = [r.confidence for r in self.reflexion_history]
        return {
            "rounds": len(self.reflexion_history),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "total_dissenting_views": sum(
                len(r.dissenting_views) for r in self.reflexion_history
            ),
            "total_open_questions": sum(
                len(r.open_questions) for r in self.reflexion_history
            ),
        }
