import re
from typing import List, Set, Dict
from collections import Counter
from .vector_store import vector_store
from ..config import settings


class SimilarityChecker:
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
    
    def extract_keywords(self, text: str) -> Set[str]:
        # Remove punctuation and lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "about",
            "your", "yours", "their", "theirs", "this", "that", "these", "those",
            "which", "who", "whom", "whose", "what", "where", "why"
        }
        return {word for word in words if word not in stop_words and len(word) > 2}
    
    def calculate_keyword_overlap(self, topic1: str, topic2: str) -> float:
        keywords1 = self.extract_keywords(topic1)
        keywords2 = self.extract_keywords(topic2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        # Use Dice coefficient for better handling of different length strings
        return (2.0 * len(intersection)) / (len(keywords1) + len(keywords2))
    
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
        keyword_sims = [self.calculate_keyword_overlap(new_topic, p) for p in past_topics]
        max_keyword_sim = max(keyword_sims) if keyword_sims else 0.0
        
        # Semantic similarity
        max_semantic_sim = self.calculate_semantic_similarity(new_topic)
        
        # Calculate combined similarity
        if settings.enable_vector_db:
            # If semantic similarity is very high, trust it more
            if max_semantic_sim > 0.8:
                max_similarity = max_semantic_sim
            else:
                # Otherwise, 60% semantic, 40% keyword
                max_similarity = (max_semantic_sim * 0.6) + (max_keyword_sim * 0.4)
        else:
            max_similarity = max_keyword_sim
        
        # Use a more balanced threshold (0.5 is usually a good 'this is too similar' point for content)
        threshold = self.similarity_threshold * 1.5 # Scaling existing 0.3 to ~0.45
        
        return {
            "max_similarity": max_similarity,
            "max_keyword_sim": max_keyword_sim,
            "max_semantic_sim": max_semantic_sim,
            "is_too_similar": max_similarity > threshold,
            "similar_topics": [
                {"topic": past_topics[i], "similarity": keyword_sims[i]}
                for i, sim in enumerate(keyword_sims)
                if sim > self.similarity_threshold
            ]
        }
    
    def calculate_novelty_score(self, topic: str, past_topics: List[str]) -> float:
        result = self.check_similarity_with_history(topic, past_topics)
        # Novelty is inverse of similarity
        novelty_score = 1.0 - result["max_similarity"]
        return max(0.0, min(1.0, novelty_score))

