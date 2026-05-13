from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory, extract_content
from ..utils.prompt_loader import prompt_loader
from ..utils.channel_profile import build_channel_context_text
from ..utils.logger import workflow_logger
import re
import json
from typing import Any, Dict, List



SCRIPT_TYPE_MODIFIERS = {
    "descriptive": (
        "Write the script in a descriptive, narrative style. "
        "Use rich imagery and storytelling. Focus on 'voice-over' feel."
    ),
    "fast-paced": (
        "Write the script in a fast-paced, high-energy style. "
        "Use short sentences, frequent pattern breaks, and quick transitions."
    ),
    "educational": (
        "Write the script in a clear, educational, and authoritative style. "
        "Focus on step-by-step implementation and key concepts."
    ),
    "story-driven": (
        "Focus on an emotional arc. Introduce a character or scenario, "
        "build tension, and provide a climax/resolution."
    ),
}

SCRIPT_TYPE_ALIASES = {
    "conversational": "fast-paced",
    "storytelling": "story-driven",
}


class ScriptAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.7, tier="heavy")
        self.system_prompt = prompt_loader.load_prompt("script_generation")

    def _apply_script_intro(self, text: str, profile: Dict) -> str:
        """Apply the global script intro line if set in profile."""
        intro = str(profile.get("script_intro_line") or "").strip()
        if not intro:
            return text

        if intro.lower() in text.lower():
            return text

        return f"{intro}\n\n{text}".strip()

    
    def _extract_script_text(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            return ""

        fenced_blocks = re.findall(r"```(?:text|markdown)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced_blocks:
            return fenced_blocks[0].strip()
        return text

    def _legacy_script_dict_to_text(self, script_data: Dict[str, Any]) -> str:
        ordered_sections = ["hook", "pattern_break", "problem", "insight", "steps", "example", "cta"]
        parts = []

        for section in ordered_sections:
            value = script_data.get(section)
            if not value:
                continue

            if section == "steps" and isinstance(value, list):
                parts.extend([str(step).strip() for step in value if str(step).strip()])
            else:
                parts.append(str(value).strip())

        return "\n\n".join(parts)

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    def _render_system_prompt(self, topic: str, category: str, style: str, channel_info: str = "") -> str:
        """Render only supported tokens and leave JSON braces untouched."""
        prompt = self.system_prompt or ""
        return (
            prompt.replace("{topic}", str(topic or ""))
            .replace("{category}", str(category or "general"))
            .replace("{style}", str(style or "descriptive"))
            .replace("{channel_info}", channel_info)
            .replace("{target_audience}", "General YouTube Audience")
        )

    def _estimate_duration_seconds(self, word_count: int) -> int:
        # Standard narration speed is ~150 words per minute (2.5 words per second)
        return int(word_count / 150 * 60)

    def _normalize_script_type(self, script_type: str | None) -> str:
        normalized = (script_type or "descriptive").strip().lower()
        normalized = SCRIPT_TYPE_ALIASES.get(normalized, normalized)
        if normalized not in SCRIPT_TYPE_MODIFIERS:
            return "descriptive"
        return normalized

    async def generate_script(
        self,
        topic: str,
        category: str = "general",
        script_type: str = "descriptive",
        channel_profile: Dict[str, Any] | None = None,
        feedback: str | None = None
    ) -> str:
        if self.llm is None:
            return (
                f"Today we break down {topic}. "
                "You will learn the problem, the hidden cause, and a practical action plan. "
                "Stay until the end for clear next steps you can apply immediately."
            )

        normalized_script_type = self._normalize_script_type(script_type)
        type_modifier = SCRIPT_TYPE_MODIFIERS.get(
            normalized_script_type, SCRIPT_TYPE_MODIFIERS["descriptive"]
        )

        if channel_profile is None:
            channel_profile = {}
        channel_profile_context = build_channel_context_text(channel_profile)

        system_content = self._render_system_prompt(
            topic=topic,
            category=category or "general",
            style=normalized_script_type,
            channel_info=channel_profile_context if channel_profile else ""
        )

        feedback_portion = f"\n\n### PREVIOUS CRITIQUE / FEEDBACK:\n{feedback}\nPlease address the points above in this new version." if feedback else ""

        human_content = (
            f"Write the full structured YouTube script now.\n\n"
            f"Topic: {topic}\n"
            f"Style: {normalized_script_type}\n"
            f"Tone/Style Details: {type_modifier}\n"
            f"{feedback_portion}"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]

        try:
            response = await llm_factory.ainvoke_with_retry(self.llm, messages)
            content = extract_content(response).strip()
        except Exception as e:
            workflow_logger.log_step(
                "generate_script",
                "warning",
                f"LLM call failed, using fallback script: {str(e)}"
            )
            return self._apply_script_intro(
                (
                    f"Today we break down {topic}. "
                    "You will learn the problem, the hidden cause, and a practical action plan. "
                    "Stay until the end for clear next steps you can apply immediately."
                ),
                channel_profile,
            )

        text = self._extract_script_text(content)
        if not text:
            text = f"Emergency fallback: This script is about {topic}. The AI output was: {content[:200]}..."
        return self._apply_script_intro(text, channel_profile)


    
    def validate_script(self, script: Any) -> Dict:
        errors = []
        warnings = []

        script_text = ""
        if isinstance(script, str):
            script_text = script.strip()
        elif isinstance(script, dict):
            # Extract narration from segments if available
            segments = script.get("segments")
            if isinstance(segments, list):
                script_text = "\n\n".join([s.get("narration", "") for s in segments if isinstance(s, dict)]).strip()
            else:
                script_text = self._legacy_script_dict_to_text(script)
        else:
            script_text = str(script).strip()

        if not script_text:
            errors.append("Script text is empty")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "word_count": 0,
                "estimated_duration_seconds": 0
            }

        word_count = self._word_count(script_text)
        if word_count < 550:
            warnings.append(f"Script is short ({word_count} words). Target is 600-800.")
        elif word_count > 900:
            warnings.append(f"Script is long ({word_count} words). Target is 600-800.")

        estimated_duration = self._estimate_duration_seconds(word_count)
        if estimated_duration < 240:
            warnings.append("Script duration is under 4 minutes.")
        elif estimated_duration > 480:
            warnings.append("Script duration is over 8 minutes.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "word_count": word_count,
            "estimated_duration_seconds": estimated_duration
        }
