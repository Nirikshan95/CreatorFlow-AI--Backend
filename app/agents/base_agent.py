from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents providing common utilities"""

    @staticmethod
    def _extract_json_candidates(content: str) -> list[str]:
        """Extract likely JSON payload candidates from free-form LLM output."""
        candidates: list[str] = []
        text = (content or "").strip()
        if not text:
            return candidates

        # 1) Prefer fenced code blocks, optionally tagged as json.
        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in fenced_blocks:
            block = block.strip()
            if block:
                candidates.append(block)

        # 2) Include the full response as a fallback.
        candidates.append(text)

        # 3) Try to extract balanced JSON object/array substrings.
        starts = [i for i, ch in enumerate(text) if ch in "{["]
        for start in starts:
            open_ch = text[start]
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        snippet = text[start : i + 1].strip()
                        if snippet:
                            candidates.append(snippet)
                        break

        # Deduplicate in order.
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    @staticmethod
    def _sanitize_json_candidate(candidate: str) -> str:
        """Normalize common model-output artifacts before JSON parsing."""
        text = (candidate or "").strip().lstrip("\ufeff")
        if not text:
            return text

        # Some models prefix with a lone "json" token.
        text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE).strip()

        # Normalize smart quotes commonly returned by chat models.
        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

        # Remove trailing commas before object/array close.
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    @staticmethod
    def parse_json(content: str) -> Optional[Any]:
        """Safely parse JSON from LLM response"""
        for candidate in BaseAgent._extract_json_candidates(content):
            try:
                return json.loads(candidate)
            except Exception:
                pass

            try:
                normalized = BaseAgent._sanitize_json_candidate(candidate)
                return json.loads(normalized)
            except Exception:
                continue

        logger.error("JSON parsing error: unable to decode model response")
        return None
