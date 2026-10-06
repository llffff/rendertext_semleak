"""Prepare a visual goal and fixed questions from a scene-text prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .backends import post_json, read_api_key


_QUOTED_TEXT = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)')

_PLANNER_INSTRUCTION = (
    "The user supplies text intended to appear in an image. Infer one checkable visual "
    "condition expressed by that text. Do not add objects, attributes, actions, or relations "
    "absent from it. Select the shortest word or phrase naming the central visual target "
    "that occurs verbatim in the supplied text; exclude presentation words such as "
    "'visible' or 'prominent'. Include it verbatim in the visual_goal. "
    'Return only JSON: {"visual_goal": "...", "word_of_interest": "..."}. '
    'If no visual condition can be inferred, return only {"error": "no_visual_goal"}.'
)

_GOAL_VALIDATION_INSTRUCTION = (
    "Check whether the proposed visual_goal accurately follows from the rendered_text, "
    "without adding or changing its visual target, attributes, actions, or relations. "
    "Use the full prompt only as context; the target must be supported by the rendered_text. "
    "Check that word_of_interest is the minimal central visual target, appears verbatim "
    "in rendered_text, and appears verbatim in visual_goal. Reject a goal that is not "
    "a checkable visual condition. "
    'Return only JSON: {"valid": true, "reason": "..."} or '
    '{"valid": false, "reason": "..."}.'
)


class GoalPlanner(Protocol):
    model: str

    def infer_goal(self, rendered_text: str) -> str: ...


@dataclass(frozen=True)
class FreeFormRequest:
    rendered_text: str
    visual_goal: str
    questions: tuple[str, str]
    goal_source: str


def freeform_questions(visual_goal: str) -> tuple[str, str]:
    """Return two visual questions for a free-form goal."""
    return (
        f"Is this visual condition present in the non-text region: {visual_goal}?",
        f"Ignoring visible written text, is there visual evidence of this condition: {visual_goal}?",
    )


class PromptPlanner:
    """Infer a candidate visual goal through an OpenAI-compatible text endpoint."""

    def __init__(self, model: str, base_url: str, *, api_key: str | None = None,
                 options: dict | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.options = options or {}

    @classmethod
    def from_vllm(cls, model: str, base_url: str) -> PromptPlanner:
        return cls(model, base_url, options={"chat_template_kwargs": {"enable_thinking": False}})

    @classmethod
    def from_openrouter(cls, model: str = "qwen/qwen3-30b-a3b", *,
                        api_key_env: str = "OPENROUTER_API_KEY",
                        env_file: str | Path = ".env") -> PromptPlanner:
        key = read_api_key(api_key_env, env_file)
        return cls(model, "https://openrouter.ai/api/v1", api_key=key,
                   options={"reasoning": {"enabled": False}})

    def _query(self, instruction: str, content: str) -> dict:
        response = post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 500,
                "temperature": 0,
                **self.options,
            },
            api_key=self.api_key,
        )
        content = response["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        try:
            result = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise ValueError("Planner did not return a JSON object") from exc
        if not isinstance(result, dict):
            raise ValueError("Planner did not return a JSON object")
        return result

    def infer_goal(self, rendered_text: str, *, prompt: str | None = None) -> str:
        proposal = self._query(_PLANNER_INSTRUCTION, rendered_text)
        if proposal.get("error") == "no_visual_goal":
            raise ValueError("No checkable visual goal was identified")
        goal, word = proposal.get("visual_goal"), proposal.get("word_of_interest")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Planner visual_goal must be a non-empty string")
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Planner word_of_interest must be a non-empty string")
        goal, word = goal.strip(), word.strip()
        if word.casefold() not in rendered_text.casefold() or word.casefold() not in goal.casefold():
            raise ValueError("Planner word_of_interest must occur in rendered_text and visual_goal")
        verdict = self._query(_GOAL_VALIDATION_INSTRUCTION, json.dumps({
            "prompt": prompt, "rendered_text": rendered_text,
            "visual_goal": goal, "word_of_interest": word,
        }, ensure_ascii=False))
        if (not isinstance(verdict.get("valid"), bool)
                or not isinstance(verdict.get("reason"), str)
                or not verdict["reason"].strip()):
            raise ValueError("Planner validation must return valid and reason")
        if not verdict["valid"]:
            raise ValueError(f"Planner visual_goal failed semantic validation: {verdict['reason']}")
        return goal


def prepare_freeform(prompt: str, *, visual_goal: str | None = None,
                     rendered_text: str | None = None,
                     planner: GoalPlanner | None = None) -> FreeFormRequest:
    """Prepare an inspectable free-form request before image scoring."""
    if rendered_text is None:
        matches = _QUOTED_TEXT.findall(prompt)
        if len(matches) != 1:
            raise ValueError("Expected one quoted rendered-text span; pass rendered_text explicitly otherwise")
        rendered_text = next(part for part in matches[0] if part)
    rendered_text = rendered_text.strip()
    if not rendered_text:
        raise ValueError("rendered_text must not be empty")
    if visual_goal is None:
        if planner is None:
            raise ValueError("Pass visual_goal or a planner to infer it from the rendered text")
        visual_goal = (planner.infer_goal(rendered_text, prompt=prompt)
                       if isinstance(planner, PromptPlanner) else planner.infer_goal(rendered_text))
        goal_source = planner.model
    else:
        goal_source = "provided"
    visual_goal = visual_goal.strip()
    if not visual_goal:
        raise ValueError("No checkable visual goal was identified; pass visual_goal explicitly")
    return FreeFormRequest(
        rendered_text=rendered_text,
        visual_goal=visual_goal,
        questions=freeform_questions(visual_goal),
        goal_source=goal_source,
    )
