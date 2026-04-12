from .base_agent import BaseAgent, HumanMessage, SystemMessage
from ..utils.llm_factory import llm_factory
from ..utils.similarity_checker import SimilarityChecker
from ..utils.prompt_loader import prompt_loader
import json
import uuid
from typing import List, Dict



class TopicAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        self.llm = llm_factory.get_llm(temperature=0.7)
        self.similarity_checker = SimilarityChecker()
        self.system_prompt = prompt_loader.load_prompt("topic_generation")

    
    def generate_topics(
        self, 
        past_topics: List[str],
        num_topics: int = 5
    ) -> List[Dict]:
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
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Generate {num_topics} fresh video topics for soft skills and communication.")
        ]
        
        response = llm_factory.invoke_with_retry(self.llm, messages)
        content = response.content
        
        topics_data = self.parse_json(content)
        if not topics_data:
            return []
            
        try:
            if isinstance(topics_data, list):
                topics = topics_data
            elif isinstance(topics_data, dict) and "topics" in topics_data:
                topics = topics_data["topics"]
            else:
                topics = [topics_data]
        except Exception:
            topics = []

        
        normalized_topics = [topic for topic in topics if isinstance(topic, dict)]

        for topic in normalized_topics:
            topic["topic_id"] = str(uuid.uuid4())
            
            if past_topics:
                novelty_score = self.similarity_checker.calculate_novelty_score(
                    topic["topic"], 
                    past_topics
                )
                topic["novelty_score"] = novelty_score
        
        return normalized_topics
    
    def select_best_topic(self, topics: List[Dict]) -> Dict:
        if not topics:
            raise ValueError("No topics provided")
        
        scored_topics = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            novelty = topic.get("novelty_score", 0.5)
            virality = topic.get("virality_score", 0.5)
            combined_score = (novelty * 0.4) + (virality * 0.6)
            topic["combined_score"] = combined_score
            scored_topics.append(topic)

        if not scored_topics:
            raise ValueError("No valid topic objects provided")
        
        scored_topics.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return scored_topics[0]
