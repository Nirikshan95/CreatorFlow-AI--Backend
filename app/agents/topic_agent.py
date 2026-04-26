from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory, extract_content
from ..utils.similarity_checker import SimilarityChecker
from ..utils.prompt_loader import prompt_loader
import json
import uuid
import re
from ..utils.logger import workflow_logger
from typing import List, Dict, Any



class TopicAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.7, tier="flash")
        self.similarity_checker = SimilarityChecker()
        self.system_prompt = prompt_loader.load_prompt("topic_generation")

    
    @staticmethod
    def _extract_topic_lines(content: str, limit: int) -> List[str]:
        text = (content or "").strip()
        if not text:
            return []

        # Prefer bullet/numbered lines first.
        candidates: List[str] = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[\).\:-]|[-*•])\s*", "", line).strip()
            if len(cleaned) >= 12:
                candidates.append(cleaned)

        if not candidates:
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            candidates = [re.sub(r"^\s*(?:\d+[\).\:-]|[-*•])\s*", "", b).strip() for b in blocks if len(b.strip()) >= 12]

        # De-duplicate while preserving order.
        seen = set()
        topics: List[str] = []
        for raw in candidates:
            normalized = re.sub(r"\s+", " ", raw).strip(" -:;,.")
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            topics.append(normalized)
            if len(topics) >= limit:
                break
        return topics

    @staticmethod
    def _derive_keywords(topic: str, max_keywords: int = 4) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", topic or "")
        stop = {
            "how", "your", "with", "that", "from", "this", "what", "when",
            "where", "which", "about", "into", "over", "under", "after", "before"
        }
        out: List[str] = []
        for word in words:
            lower = word.lower()
            if lower in stop:
                continue
            out.append(word)
            if len(out) >= max_keywords:
                break
        return out

    async def generate_topics(
        self, 
        past_topics: List[str],
        num_topics: int = 5,
        category: str | None = None,
        channel_profile: Dict | None = None
    ) -> List[Dict]:
        def _default_topics() -> List[Dict]:
            fallback_items = [
                "How to build confidence and overcome self-doubt",
                "Why overthinking kills momentum and how to stop",
                "Daily habits that rebuild self-discipline fast",
                "How to stay consistent when motivation disappears",
                "The confidence framework top performers use",
            ]
            out: List[Dict] = []
            target = max(1, num_topics)
            for line in fallback_items[:target]:
                out.append({
                    "topic": line,
                    "novelty_score": 0.8,
                    "virality_score": 0.75,
                    "category": category or "general",
                    "keywords": self._derive_keywords(line),
                    "rationale": "Fallback topic generated due to LLM response failure."
                })
            return out

        if self.llm is None:
            return [
                {
                    "topic_id": str(uuid.uuid4()),
                    "topic": "How to overcome public speaking anxiety",
                    "novelty_score": 0.9,
                    "virality_score": 0.8,
                    "category": "confidence",
                    "keywords": ["public speaking", "anxiety", "confidence"],
                    "rationale": "High demand topic with viral potential"
                }
            ]
        
        past_topics_text = "\n".join(f"- {topic}" for topic in past_topics[:30]) or "- None"
        category_hint = category or "general trending interest"
        rendered_system_prompt = (self.system_prompt or "").replace("{num_topics}", str(num_topics))
        
        from ..utils.channel_profile import build_channel_context_text
        channel_context = build_channel_context_text(channel_profile or {})

        messages = [
            SystemMessage(content=rendered_system_prompt),
            HumanMessage(content=(
                f"Generate exactly {num_topics} fresh video topics.\n"
                f"Category preference: {category_hint}\n\n"
                f"{channel_context}\n\n"
                "Previously covered topics to avoid repeating:\n"
                f"{past_topics_text}"
            ))
        ]
        
        content = ""
        try:
            response = await llm_factory.ainvoke_with_retry(self.llm, messages)
            content = extract_content(response).strip()
        except Exception as e:
            workflow_logger.log_step(
                "generate_topics",
                "warning",
                f"LLM call failed, using fallback topics: {str(e)}"
            )
            topics = _default_topics()
            normalized_topics = [topic for topic in topics if isinstance(topic, dict) and topic.get("topic")]
            for topic in normalized_topics:
                topic["topic_id"] = str(uuid.uuid4())
                if past_topics:
                    novelty_score = self.similarity_checker.calculate_novelty_score(topic["topic"], past_topics)
                    topic["novelty_score"] = (topic["novelty_score"] * 0.4) + (novelty_score * 0.6)
            normalized_topics.sort(key=lambda t: (t.get("novelty_score", 0), t.get("virality_score", 0)), reverse=True)
            return normalized_topics[:num_topics]
        
        topics: List[Dict] = []
        
        # Try JSON parsing first
        try:
            # Clean possible markdown fences
            clean_content = re.sub(r"```json\s*|\s*```", "", content).strip()
            data = json.loads(clean_content)
            if isinstance(data, dict) and "topics" in data:
                for t in data["topics"]:
                    if isinstance(t, dict) and t.get("topic"):
                        topics.append({
                            "topic": t.get("topic"),
                            "novelty_score": float(t.get("novelty_score", 0.8)),
                            "virality_score": float(t.get("virality_score", 0.75)),
                            "category": t.get("category") or category or "general",
                            "keywords": t.get("keywords") or self._derive_keywords(t["topic"]),
                            "rationale": t.get("rationale") or "Generated from structured synthesis."
                        })
        except (json.JSONDecodeError, ValueError):
            # Fallback to text-based extraction
            extracted = self._extract_topic_lines(content, num_topics)
            for line in extracted:
                topics.append({
                    "topic": line,
                    "novelty_score": 0.8,
                    "virality_score": 0.75,
                    "category": category or "general",
                    "keywords": self._derive_keywords(line),
                    "rationale": "Generated from text-based extraction (JSON parse failed)."
                })

        normalized_topics = [topic for topic in topics if isinstance(topic, dict) and topic.get("topic")]
        if not normalized_topics:
            workflow_logger.log_step(
                "generate_topics",
                "warning",
                "No valid topics parsed from model output, using fallback topics."
            )
            normalized_topics = _default_topics()

        for topic in normalized_topics:
            topic["topic_id"] = str(uuid.uuid4())
            workflow_logger.log_step("topic_analysis", "info", f"Analyzing topic novelty: {topic['topic']}")
            
            # Re-calculate novelty based on similarity if past topics exist
            if past_topics:
                novelty_score = self.similarity_checker.calculate_novelty_score(
                    topic["topic"], 
                    past_topics
                )
                # Blend LLM and Similarity scores
                topic["novelty_score"] = (topic["novelty_score"] * 0.4) + (novelty_score * 0.6)
        
        return normalized_topics[:num_topics]
    
    async def select_best_topic(
        self, 
        topics: List[Dict], 
        category: str = "general",
        channel_profile: Dict[str, Any] | None = None
    ) -> Dict:
        if not topics:
            raise ValueError("No topics provided")
        
        # 1. Quick scoring for fallback/pre-sorting
        scored_topics = []
        for topic in topics:
            if not isinstance(topic, dict): continue
            novelty = topic.get("novelty_score", 0.5)
            virality = topic.get("virality_score", 0.5)
            topic["selection_score"] = (novelty * 0.4) + (virality * 0.6)
            scored_topics.append(topic)

        if not scored_topics:
            raise ValueError("No valid topic objects provided")

        # 2. Strategic LLM Selection
        if self.llm:
            try:
                from ..utils.channel_profile import build_channel_context_text
                channel_profile_context = build_channel_context_text(channel_profile or {})
                topic_list_text = "\n".join([f"{i+1}. {t['topic']} (Rationale: {t.get('rationale', 'N/A')})" for i, t in enumerate(scored_topics)])
                
                messages = [
                    SystemMessage(content=(
                        "You are a growth-focused YouTube Content Strategist. "
                        "Your job is to pick the SINGLE MOST VIRAL topic from a list."
                    )),
                    HumanMessage(content=(
                        f"Select the absolute best topic for our next video.\n\n"
                        f"Category: {category}\n"
                        f"Channel Context:\n{channel_profile_context}\n\n"
                        f"Candidate Topics:\n{topic_list_text}\n\n"
                        "Criteria:\n"
                        "1. High retention potential\n"
                        "2. Fits the channel voice\n"
                        "3. Strong hook promise\n\n"
                        f"Return ONLY the number (1-{len(scored_topics)}) of the winning topic."
                    ))
                ]
                
                response = await llm_factory.ainvoke_with_retry(self.llm, messages)
                match = re.search(r"(\d+)", extract_content(response))
                if match:
                    index = int(match.group(1)) - 1
                    if 0 <= index < len(scored_topics):
                        return scored_topics[index]
            except Exception:
                pass # Fallback to logic

        # Fallback: Highest score
        scored_topics.sort(key=lambda x: x["selection_score"], reverse=True)
        return scored_topics[0]
