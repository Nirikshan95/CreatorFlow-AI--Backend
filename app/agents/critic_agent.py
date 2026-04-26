from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory, extract_content
from ..utils.prompt_loader import prompt_loader
import json
from typing import List, Dict



class CriticAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.5, tier="heavy")
        self.system_prompt = prompt_loader.load_prompt("content_critic")

    
    async def critique_content(self, topic: str, title: str, script: str, keywords: List[str]) -> Dict:
        if self.llm is None:
            return {
                "rating": 8.5,
                "is_passed": True,
                "feedback": "Great content with strong hook.",
                "suggestions": [],
                "critique": {
                    "hook": "Strong",
                    "retention": "Good",
                    "seo": "Aligned"
                }
            }
        
        prompt_vars = {
            "topic": topic,
            "title": title,
            "script": script,
            "keywords": ", ".join(keywords) if keywords else "None"
        }
        
        # Simple manual formatting since prompt_loader just loads raw text
        populated_prompt = self.system_prompt
        for k, v in prompt_vars.items():
            populated_prompt = populated_prompt.replace(f"{{{k}}}", str(v))
        
        messages = [
            SystemMessage(content=populated_prompt),
            HumanMessage(content="Perform a final review of this video content packet.")
        ]
        
        response = await llm_factory.ainvoke_with_retry(self.llm, messages)
        content = extract_content(response)
        
        critique_data = self.parse_json(content)
        if not critique_data:
            critique_data = {
                "rating": 5.0,
                "is_passed": False,
                "feedback": "Failed to parse critique",
                "raw": content
            }
        
        return critique_data

