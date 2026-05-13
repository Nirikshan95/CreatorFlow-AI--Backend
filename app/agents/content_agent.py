from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory, extract_content
from ..utils.prompt_loader import prompt_loader
from ..utils.channel_profile import build_channel_context_text
import re
from ..utils.logger import workflow_logger
from typing import Dict, Any, List


class ContentAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.8, tier="flash")
        self.community_prompt = prompt_loader.load_prompt("community_post")
        self.thumbnail_prompt = prompt_loader.load_prompt("thumbnail_prompt")
        self.marketing_prompt = prompt_loader.load_prompt("marketing_strategy")
        self.post_image_prompt = prompt_loader.load_prompt("post_image_prompt")

    @staticmethod
    def _split_suggestion_text(value: Any) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []

        normalized = text.replace("\r\n", "\n")
        normalized = re.sub(r"[•●▪◦]", "\n- ", normalized)

        by_lines = [
            re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", line).strip()
            for line in re.split(r"\n+", normalized)
        ]
        by_lines = [line for line in by_lines if line]
        if len(by_lines) > 1:
            return by_lines

        by_numbering = [
            re.sub(r"^\s*\d+[.)]\s+", "", part).strip()
            for part in re.split(r"\s+(?=\d+[.)]\s+)", normalized)
        ]
        by_numbering = [part for part in by_numbering if part]
        if len(by_numbering) > 1:
            return by_numbering

        return [text]

    @classmethod
    def _normalize_suggestions(cls, raw: Any) -> List[str]:
        if raw is None:
            return []

        items: List[str] = []
        if isinstance(raw, list):
            for entry in raw:
                items.extend(cls._split_suggestion_text(entry))
        else:
            items.extend(cls._split_suggestion_text(raw))

        deduped: List[str] = []
        seen = set()
        for item in items:
            clean = str(item).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clean)

        return deduped[:8]

    @staticmethod
    def _normalize_image_prompt_intent(prompt: str) -> str:
        text = re.sub(r"\s+", " ", str(prompt or "")).strip().strip('"')
        if not text:
            return ""

        allowed_starts = (
            "generate an image of",
            "create a realistic image of",
            "professional photography of",
        )
        lowered = text.lower()
        if any(lowered.startswith(start) for start in allowed_starts):
            return text

        # Remove common leading labels before enforcing intent prefix.
        text = re.sub(
            r"^(?:thumbnail prompt|post image prompt|image prompt|prompt)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return f"Generate an image of {text}".strip()

    async def generate_community_posts(
        self,
        topic: str,
        script: Any,
        title: str = "",
        hashtags: List[str] | None = None,
        category: str | None = None,
        channel_profile: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "post_type": "post",
                "content": f"New video is live: {topic}. Watch now and share your take."
            }

        script_text = self.compact_script_context(script, max_chars=1800)

        if channel_profile is None:
            channel_profile = {}
        channel_profile_context = build_channel_context_text(channel_profile)

        final_prompt = (
            self.community_prompt
            .replace("{topic}", topic)
            .replace("{title}", title)
            .replace("{category}", category or "General")
            .replace("{hashtags}", ", ".join(hashtags or []))
            .replace("{script}", script_text)
            .replace("{channel_info}", channel_profile_context)
        )

        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=(
                f"Topic: {topic}\n"
                f"Title: {title}\n"
                f"Category: {category or 'General'}\n\n"
                f"Please generate the community post now."
            ))
        ]

        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response).strip()

        data = self.parse_json(content)
        if not isinstance(data, dict):
            return {
                "post_type": "post",
                "content": (content or "").strip()
            }

        post_type = str(data.get("post_type") or "post").strip().lower()
        if post_type == "poll":
            options = data.get("options")
            if not isinstance(options, list):
                options = []
            options = [str(opt).strip() for opt in options if str(opt).strip()][:4]
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")
            return {
                "post_type": "poll",
                "poll_question": str(data.get("poll_question") or f"What is your biggest challenge with {topic}?").strip(),
                "options": options
            }

        return {
            "post_type": "post",
            "content": str(data.get("content") or "").strip()
        }

    async def generate_post_image_prompt(self, topic: str, script: Any, post_text: str) -> Dict[str, str]:
        if self.llm is None:
            return {"post_image_prompt": f"Social post image for {topic}"}

        script_text = self.compact_script_context(script, max_chars=1400)

        final_prompt = (
            self.post_image_prompt
            .replace("{topic}", topic)
            .replace("{script}", script_text)
            .replace("{post_text}", post_text)
        )

        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=f"Please generate the post image prompt for this post: {post_text}")
        ]

        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response).strip()

        data = self.parse_json(content)
        if isinstance(data, dict) and isinstance(data.get("post_image_prompt"), str):
            return {"post_image_prompt": self._normalize_image_prompt_intent(data["post_image_prompt"])}

        return {"post_image_prompt": self._normalize_image_prompt_intent(content)}

    async def generate_thumbnail_prompts(
        self,
        topic: str,
        title: str,
        script: Any,
        category: str | None = None,
        channel_profile: Dict[str, Any] | None = None
    ) -> Dict[str, str]:
        if self.llm is None:
            return {"thumbnail_prompt": f"YouTube thumbnail prompt for {topic} with title {title}"}

        script_text = self.compact_script_context(script, max_chars=1400)

        if channel_profile is None:
            channel_profile = {}
        channel_profile_context = build_channel_context_text(channel_profile)

        final_prompt = self.thumbnail_prompt.replace("{topic}", topic).replace("{title}", title).replace("{script}", script_text)

        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=(
                f"Topic: {topic}\n"
                f"Category: {category or 'General'}\n"
                f"Title: {title}\n\n"
                f"{channel_profile_context}"
            ))
        ]

        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response).strip()

        data = self.parse_json(content)
        if isinstance(data, dict) and isinstance(data.get("thumbnail_prompt"), str):
            return {"thumbnail_prompt": self._normalize_image_prompt_intent(data["thumbnail_prompt"])}

        return {"thumbnail_prompt": self._normalize_image_prompt_intent(content)}

    async def generate_marketing_strategy(
        self,
        topic: str,
        script: Any,
        title: str = "",
        category: str | None = None,
        channel_profile: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {"suggestions": []}

        script_text = self.compact_script_context(script, max_chars=2200)

        if channel_profile is None:
            channel_profile = {}
        channel_profile_context = build_channel_context_text(channel_profile)

        final_prompt = self.marketing_prompt.replace("{topic}", topic).replace("{script}", script_text)

        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=(
                f"Topic: {topic}\n"
                f"Category: {category or 'General'}\n"
                f"Title: {title}\n\n"
                f"{channel_profile_context}\n\n"
                "Return ONLY valid JSON."
            ))
        ]

        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response).strip()

        data = self.parse_json(content)
        if isinstance(data, dict):
            clean_suggestions = self._normalize_suggestions(data.get("suggestions"))
            if clean_suggestions:
                workflow_logger.log_step("marketing_strategy", "success", "Parsed distribution suggestions")
                return {"suggestions": clean_suggestions}
        elif isinstance(data, list):
            clean_suggestions = self._normalize_suggestions(data)
            if clean_suggestions:
                workflow_logger.log_step("marketing_strategy", "success", "Parsed distribution suggestions from array payload")
                return {"suggestions": clean_suggestions}

        clean_suggestions = self._normalize_suggestions(content)
        if clean_suggestions:
            workflow_logger.log_step("marketing_strategy", "warning", "Used text fallback normalization for suggestions")
            return {"suggestions": clean_suggestions}

        workflow_logger.log_step("marketing_strategy", "warning", "Failed to parse suggestions JSON, using minimal fallback")
        return {
            "suggestions": [
                f"Publish a Shorts teaser for {topic} with a strong hook and clear CTA.",
                "Share the video in one relevant community where your audience is already active.",
                "Pin a community post with a direct question to drive early comments.",
            ]
        }
