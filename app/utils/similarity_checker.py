from typing import List, Set, Dict
from collections import Counter
from .vector_store import vector_store
from ..config import settings


class SimilarityChecker:
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
    
    def extract_keywords(self, text: str) -> Set[str]:
        words = text.lower().split()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "about"
        }
        return {word for word in words if word not in stop_words and len(word) > 2}
    
    def calculate_keyword_overlap(self, topic1: str, topic2: str) -> float:
        keywords1 = self.extract_keywords(topic1)
        keywords2 = self.extract_keywords(topic2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_semantic_similarity(self, new_topic: str) -> float:
        """Calculate maximum semantic similarity with past topics in VectorStore"""
        if not settings.enable_vector_db:
            return 0.0
            
        similar_topics = vector_store.find_similar_topics(new_topic, k=1)
        if not similar_topics:
            return 0.0
            
        return similar_topics[0]["similarity"]

    def check_similarity_with_history(
        self, 
        new_topic: str, 
        past_topics: List[str]
    ) -> Dict:
        # Keyword-based similarity
        keyword_similarities = []
        for past_topic in past_topics:
            similarity = self.calculate_keyword_overlap(new_topic, past_topic)
            keyword_similarities.append(similarity)
        
        max_keyword_sim = max(keyword_similarities) if keyword_similarities else 0.0
        
        # Semantic similarity
        max_semantic_sim = self.calculate_semantic_similarity(new_topic)
        
        # Combine similarities (prioritize semantic if enabled)
        if settings.enable_vector_db:
            # Weighted average: 70% semantic, 30% keyword
            max_similarity = (max_semantic_sim * 0.7) + (max_keyword_sim * 0.3)
        else:
            max_similarity = max_keyword_sim
        
        return {
            "max_similarity": max_similarity,
            "max_keyword_sim": max_keyword_sim,
            "max_semantic_sim": max_semantic_sim,
            "is_too_similar": max_similarity > self.similarity_threshold,
            "similar_topics": [
                {"topic": past_topics[i], "similarity": keyword_similarities[i]}
                for i in range(len(past_topics))
                if keyword_similarities[i] > self.similarity_threshold
            ]
        }
    
    def calculate_novelty_score(self, topic: str, past_topics: List[str]) -> float:
        result = self.check_similarity_with_history(topic, past_topics)
        max_similarity = result["max_similarity"]
        novelty_score = 1.0 - max_similarity
        return max(0.0, min(1.0, novelty_score))

