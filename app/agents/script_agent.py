from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory
from ..utils.prompt_loader import prompt_loader
import re
from typing import Any, Dict



class ScriptAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.8)
        self.system_prompt = prompt_loader.load_prompt("script_generation")

    
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

        if parts:
            return "\n\n".join(parts).strip()

        script_text = script_data.get("script")
        if isinstance(script_text, str):
            return script_text.strip()

        raw_content = script_data.get("raw_content")
        if isinstance(raw_content, str):
            return raw_content.strip()

        return ""

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b[\w']+\b", text or ""))

    @staticmethod
    def _estimate_duration_seconds(word_count: int) -> int:
        # 145-160 WPM is common for YouTube narration.
        return max(1, int(round((word_count / 150) * 60)))

    def generate_script(self, topic: str) -> str:
        if self.llm is None:
            return (
                "You do not fail interviews because you lack skill. You fail because nerves steal your voice.\n\n"
                "Here is the twist. Confidence is not a personality trait. It is a repeatable system.\n\n"
                "Most people over-prepare answers and under-prepare delivery. That is the real problem.\n\n"
                "Use this method. First, record two mock answers daily for seven days. "
                "Second, pause two seconds before each answer to control pace. "
                "Third, replace vague claims with one proof line from your work. "
                "Fourth, end every answer with a result and a number. "
                "Fifth, practice one hard question first, not last.\n\n"
                "One candidate I coached was rejected four times. "
                "She used this system for ten days. "
                "On her fifth interview, she stayed calm and clear. "
                "She got the offer in forty eight hours.\n\n"
                "If you want more scripts like this, follow and practice today."
            )
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Generate a video script for this topic: {topic}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        script_data = self.parse_json(content)
        if isinstance(script_data, dict):
            text = self._legacy_script_dict_to_text(script_data)
            if text:
                return text

        text = self._extract_script_text(str(content))
        return text

    
    def validate_script(self, script: Any) -> Dict:
        errors = []
        warnings = []

        script_text = ""
        if isinstance(script, str):
            script_text = script.strip()
        elif isinstance(script, dict):
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
        if word_count < 300:
            warnings.append("Script is too short (minimum 300 words)")
        elif word_count > 450:
            warnings.append("Script is too long (maximum 450 words)")

        estimated_duration = self._estimate_duration_seconds(word_count)
        if estimated_duration < 120:
            warnings.append("Script is too short (minimum 2 minutes)")
        elif estimated_duration > 180:
            warnings.append("Script is too long (maximum 3 minutes)")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "word_count": word_count,
            "estimated_duration_seconds": estimated_duration
        }
