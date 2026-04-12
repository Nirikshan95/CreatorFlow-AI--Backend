from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory
from ..utils.prompt_loader import prompt_loader
import json
from typing import List, Dict



class ContentAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.8)
        self.community_prompt = prompt_loader.load_prompt("community_post")
        self.thumbnail_prompt = prompt_loader.load_prompt("thumbnail_prompt")
        self.marketing_prompt = prompt_loader.load_prompt("marketing_strategy")

    
    def generate_community_posts(self, topic: str) -> List[Dict]:
        if self.llm is None:
            return [
                {
                    "post_text": "How many of you still get nervous before speaking? I used to freeze up every time. Here's what changed everything...",
                    "character_count": 145,
                    "engagement_trigger": "question",
                    "question_asked": "How many of you still get nervous before speaking?",
                    "tone": "casual",
                    "emoji_count": 0
                }
            ]
        
        messages = [
            SystemMessage(content=self.community_prompt),
            HumanMessage(content=f"Generate 3 community posts for a video about: {topic}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        posts_data = self.parse_json(content)
        if not posts_data:
            return []
            
        try:
            if isinstance(posts_data, list):
                posts = posts_data
            elif isinstance(posts_data, dict) and "posts" in posts_data:
                posts = posts_data["posts"]
            else:
                posts = [posts_data]
        except Exception:
            posts = []

        
        return posts
    
    def generate_thumbnail_prompts(self, topic: str, title: str) -> List[Dict]:
        if self.llm is None:
            return [
                {
                    "subject": "Professional person with shocked expression",
                    "expression": "Shocked and surprised",
                    "color_scheme": "Yellow background with blue text",
                    "text_overlay": "Stop Being Nervous",
                    "background": "Solid bright yellow",
                    "style_reference": "YouTube thumbnail style",
                    "full_prompt": "Professional person with shocked expression, solid bright yellow background, blue text overlay saying 'Stop Being Nervous', YouTube thumbnail style, high contrast",
                    "ctr_prediction": "high"
                }
            ]
        
        messages = [
            SystemMessage(content=self.thumbnail_prompt),
            HumanMessage(content=f"Generate 2 thumbnail prompts for:\nTopic: {topic}\nTitle: {title}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        prompts_data = self.parse_json(content)
        if not prompts_data:
            return []
            
        try:
            if isinstance(prompts_data, list):
                prompts = prompts_data
            elif isinstance(prompts_data, dict) and "prompts" in prompts_data:
                prompts = prompts_data["prompts"]
            else:
                prompts = [prompts_data]
        except Exception:
            prompts = []

        
        return prompts
    
    def generate_marketing_strategy(self, topic: str) -> Dict:
        if self.llm is None:
            return {
                "distribution_channels": [
                    {
                        "platform": "WhatsApp",
                        "action": "Share in 3-5 relevant soft skills groups",
                        "timing": "Within 1 hour of upload"
                    },
                    {
                        "platform": "LinkedIn",
                        "action": "Post with key insights from video",
                        "timing": "Within 2 hours of upload"
                    }
                ],
                "early_engagement_hack": "Comment on your own video within first 10 minutes asking a specific question",
                "repurposing_ideas": [
                    "Create 3 Shorts from key points",
                    "Extract quotes for Twitter/X",
                    "Make Instagram carousel"
                ],
                "community_outreach": [
                    "Share in Reddit r/publicspeaking",
                    "Post in Toastmasters groups",
                    "Share in communication skills communities"
                ],
                "first_1000_views_strategy": "1. Share in 5 WhatsApp groups immediately after upload\n2. Post on LinkedIn with personal story\n3. Create 3 Shorts and schedule them\n4. Comment within first 10 minutes\n5. Reply to every comment in first hour"
            }
        
        messages = [
            SystemMessage(content=self.marketing_prompt),
            HumanMessage(content=f"Generate a marketing strategy for a video about: {topic}")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        strategy_data = self.parse_json(content)
        if not strategy_data:
            strategy_data = {
                "error": "Failed to parse marketing strategy",
                "raw_content": content
            }
        
        return strategy_data

