from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory, extract_content
from ..utils.prompt_loader import prompt_loader
from ..utils.channel_profile import build_channel_context_text
import re
from ..utils.logger import workflow_logger
from typing import List, Dict, Any


class SEOAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.7, tier="flash")
        self.seo_prompt = prompt_loader.load_prompt("seo_package")

    @staticmethod
    def _apply_channel_defaults(description: str, profile: Dict) -> str:
        text = (description or "").strip()
        if not text:
            return text

        intro_line = str(profile.get("intro_line") or "").strip()
        if intro_line and intro_line.lower() not in text.lower():
            text = f"{intro_line}\n\n{text}"

        useful_links_raw = profile.get("useful_links") or []
        useful_links_formatted = []
        for item in useful_links_raw:
            if isinstance(item, dict):
                key = str(item.get("key", "")).strip()
                value = str(item.get("value", "")).strip()
                if key and value:
                    useful_links_formatted.append(f"{key}: {value}")
            elif isinstance(item, str) and item.strip():
                useful_links_formatted.append(item.strip())

        if useful_links_formatted:
            missing = [link for link in useful_links_formatted if link.lower() not in text.lower()]
            if missing:
                links_block = "Useful Links:\n" + "\n".join(f"- {link}" for link in missing)
                text = f"{text}\n\n{links_block}"

        social_links = profile.get("social_links") or []
        if social_links:
            social_links_text = ", ".join(str(link).strip() for link in social_links if str(link).strip())
            if social_links_text and social_links_text.lower() not in text.lower():
                text = f"{text}\n\nFollow me: {social_links_text}"

        footer = str(profile.get("description_footer") or "").strip()
        if footer and footer.lower() not in text.lower():
            text = f"{text}\n\n{footer}"

        return text.strip()

    @staticmethod
    def _looks_like_json_blob(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        if raw.startswith("{") or raw.startswith("["):
            return True
        lowered = raw.lower()
        return '"video_title"' in lowered or '"description"' in lowered or '"hashtags"' in lowered

    def _render_system_prompt(self, topic: str, category: str, title: str, keywords: List[str], script: str, channel_info: str, feedback: str) -> str:
        prompt = self.seo_prompt or ""
        return (
            prompt.replace("{topic}", str(topic or ""))
            .replace("{category}", str(category or "General"))
            .replace("{title}", str(title or ""))
            .replace("{keywords}", ", ".join(keywords) if keywords else "")
            .replace("{script}", self.compact_script_context(script, max_chars=1800))
            .replace("{channel_info}", str(channel_info or "No specific channel context."))
            .replace("{feedback}", str(feedback or "No previous feedback."))
        )

    async def generate_seo_package(
        self,
        topic: str,
        title: str,
        keywords: List[str],
        category: str | None = None,
        channel_profile: Dict | None = None,
        feedback: str | None = None,
        script: str | None = None
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "video_title": title or f"The Ultimate Guide to {topic}",
                "description": f"In this video, we break down {topic} with clear examples and practical steps.",
                "hashtags": ["#youtube", "#growth", "#learning"]
            }

        if channel_profile is None:
            channel_profile = {}
        channel_profile_context = build_channel_context_text(channel_profile)

        system_content = self._render_system_prompt(
            topic=topic,
            category=category,
            title=title,
            keywords=keywords,
            script=script,
            channel_info=channel_profile_context,
            feedback=feedback
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content="Please generate the SEO package now.")
        ]

        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response).strip()

        data = self.parse_json(content)
        if not isinstance(data, dict):
            fallback_description = f"In this video, we break down {topic} with clear examples and practical steps."
            if content and not self._looks_like_json_blob(content):
                fallback_description = (content or "").strip()
            return {
                "video_title": title or f"About {topic}",
                "description": self._apply_channel_defaults(fallback_description, channel_profile),
                "hashtags": []
            }

        video_title = str(data.get("video_title") or title or f"About {topic}").strip()
        description = self._apply_channel_defaults(str(data.get("description") or "").strip(), channel_profile)

        hashtags = data.get("hashtags")
        if not isinstance(hashtags, list):
            hashtags = []

        clean_hashtags = []
        seen_tags = set()
        for tag in hashtags:
            clean = str(tag or "").strip()
            if not clean:
                continue
            if not clean.startswith("#"):
                clean = f"#{clean.lstrip('#')}"
            clean = clean.replace(" ", "")
            key = clean.lower()
            if key in seen_tags:
                continue
            seen_tags.add(key)
            clean_hashtags.append(clean)

        for tag in channel_profile.get("default_hashtags") or []:
            clean = str(tag).strip()
            if not clean:
                continue
            if not clean.startswith("#"):
                clean = f"#{clean.lstrip('#')}"
            clean = clean.replace(" ", "")
            key = clean.lower()
            if key in seen_tags:
                continue
            seen_tags.add(key)
            clean_hashtags.append(clean)

        workflow_logger.log_step("generate_seo", "success", "Generated SEO package")
        return {
            "video_title": video_title,
            "description": description,
            "hashtags": clean_hashtags
        }

    async def select_best_title(
        self,
        titles: List[Dict],
        topic: str = "",
        category: str = "general",
        channel_profile: Dict[str, Any] | None = None
    ) -> Dict:
        if not titles:
            raise ValueError("No titles provided")

        normalized_titles = [t for t in titles if isinstance(t, dict) and t.get("title")]
        if not normalized_titles:
            raise ValueError("No valid title objects provided")

        scored_titles = []
        for t in normalized_titles:
            ctr_raw = t.get("ctr_prediction", 0.5)
            score = float(ctr_raw) if isinstance(ctr_raw, (int, float)) else 0.7

            length = t.get("character_count", 0)
            if length < 40:
                score *= 0.6
            elif length > 75:
                score *= 0.7
            elif 50 <= length <= 70:
                score *= 1.2

            t["selection_score"] = score
            scored_titles.append(t)

        if self.llm:
            try:
                channel_profile_context = build_channel_context_text(channel_profile or {})
                titles_text = "\n".join([f"{i+1}. {t['title']} (Angle: {t.get('pattern_used', 'N/A')})" for i, t in enumerate(scored_titles)])

                messages = [
                    SystemMessage(content="You are a YouTube Title specialist."),
                    HumanMessage(content=(
                        f"Pick the SINGLE best title for this video topic: {topic}\n\n"
                        f"Category: {category}\n"
                        f"Channel Background:\n{channel_profile_context}\n\n"
                        f"Candidates:\n{titles_text}\n\n"
                        "Return ONLY the number of the winning title."
                    ))
                ]

                response = await llm_factory.ainvoke_with_retry(self.llm, messages)
                match = re.search(r"(\d+)", extract_content(response))
                if match:
                    index = int(match.group(1)) - 1
                    if 0 <= index < len(scored_titles):
                        return scored_titles[index]
            except Exception:
                pass

        scored_titles.sort(key=lambda x: x["selection_score"], reverse=True)
        return scored_titles[0]
