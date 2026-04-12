from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory
from ..utils.prompt_loader import prompt_loader
import re
from typing import List, Dict



class SEOAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.7)
        self.title_prompt = prompt_loader.load_prompt("seo_title")
        self.description_prompt = prompt_loader.load_prompt("seo_description")

    
    def generate_titles(self, topic: str, keywords: List[str]) -> List[Dict]:
        if self.llm is None:
            return [
                {
                    "title": "Stop Being Nervous: 3 Confidence Hacks",
                    "character_count": 42,
                    "pattern_used": "contrarian",
                    "primary_keyword": "confidence",
                    "ctr_prediction": "high",
                    "rationale": "Strong hook with clear benefit"
                }
            ]
        
        messages = [
            SystemMessage(content=self.title_prompt),
            HumanMessage(content=f"Generate 3 optimized titles for a video about: {topic}\nKeywords: {', '.join(keywords)}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        titles_data = self.parse_json(content)
        if not titles_data:
            return []
            
        try:
            if isinstance(titles_data, list):
                titles = titles_data
            elif isinstance(titles_data, dict) and "titles" in titles_data:
                titles = titles_data["titles"]
            else:
                titles = [titles_data]
        except Exception:
            titles = []

        
        return titles
    
    def generate_description(
        self, 
        topic: str, 
        title: str, 
        keywords: List[str]
    ) -> str:
        def extract_text(content: str) -> str:
            text = (content or "").strip()
            if not text:
                return ""
            fenced_blocks = re.findall(r"```(?:text|markdown)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
            if fenced_blocks:
                return fenced_blocks[0].strip()
            return text

        def legacy_description_to_text(data: Dict) -> str:
            full = data.get("full_description")
            if isinstance(full, str) and full.strip():
                return full.strip()

            parts = []
            first_two_lines = data.get("first_two_lines")
            video_summary = data.get("video_summary")
            key_points = data.get("key_points")
            cta = data.get("cta")
            hashtags = data.get("hashtags")

            if isinstance(first_two_lines, str) and first_two_lines.strip():
                parts.append(first_two_lines.strip())
            if isinstance(video_summary, str) and video_summary.strip():
                parts.append(video_summary.strip())
            if isinstance(key_points, list):
                key_points_text = " ".join([str(p).strip() for p in key_points if str(p).strip()])
                if key_points_text:
                    parts.append(key_points_text)
            if isinstance(cta, str) and cta.strip():
                parts.append(cta.strip())
            if isinstance(hashtags, list):
                tags = " ".join([str(h).strip() for h in hashtags if str(h).strip()])
                if tags:
                    parts.append(tags)

            return "\n\n".join(parts).strip()

        if self.llm is None:
            return (
                "Confidence is not magic. It is built through daily action, not lucky personality.\n\n"
                "In this video, you will learn simple speaking techniques that calm nerves fast and make your words land with power. "
                "Use these methods before interviews, meetings, and presentations when pressure feels highest.\n\n"
                "Watch till the end, pick one technique, and try it today. "
                "Tell me in the comments which moment usually breaks your confidence. "
                "Like, subscribe, and share this with someone who needs a speaking win this week.\n\n"
                "#confidence #publicspeaking #communication #selfimprovement"
            )
        
        messages = [
            SystemMessage(content=self.description_prompt),
            HumanMessage(content=f"Generate an optimized description for:\nTopic: {topic}\nTitle: {title}\nKeywords: {', '.join(keywords)}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        description_data = self.parse_json(content)
        if isinstance(description_data, dict):
            legacy_text = legacy_description_to_text(description_data)
            if legacy_text:
                return legacy_text

        return extract_text(str(content))

    
    def select_best_title(self, titles: List[Dict]) -> Dict:
        if not titles:
            raise ValueError("No titles provided")
        
        normalized_titles = [t for t in titles if isinstance(t, dict)]
        if not normalized_titles:
            raise ValueError("No valid title objects provided")

        valid_titles = [t for t in normalized_titles if 50 <= t.get("character_count", 0) <= 70]
        
        if valid_titles:
            return valid_titles[0]
        elif normalized_titles:
            return normalized_titles[0]
        else:
            raise ValueError("No valid titles available")
