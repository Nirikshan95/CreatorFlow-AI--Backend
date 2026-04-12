from typing import TypedDict, List, Dict, Optional, Annotated, Any
import operator
from pydantic import BaseModel


def merge_dict(a: Dict, b: Dict) -> Dict:
    """Simple dictionary merge reducer"""
    return {**a, **b}


class ContentState(TypedDict):
    """State for the content generation workflow"""
    past_topics: Annotated[List[str], operator.add]

    num_topics: int
    category: Optional[str]
    
    # Topic generation
    generated_topics: List[Dict]
    selected_topic: Optional[Dict]
    
    # Script generation
    script: Optional[Any]
    script_validation: Optional[Dict]
    
    # SEO generation
    titles: List[Dict]
    selected_title: Optional[Dict]
    description: Optional[Any]
    
    # Content generation
    community_posts: List[Dict]
    selected_community_post: Optional[Dict]
    thumbnail_prompts: List[Dict]
    selected_thumbnail_prompt: Optional[Dict]
    marketing_strategy: Optional[Dict]
    
    # Final output
    final_content: Optional[Dict]
    
    # Error handling
    errors: Annotated[List[str], operator.add]
    retries: Annotated[Dict[str, int], merge_dict]

    max_retries: int


class WorkflowConfig(BaseModel):
    """Configuration for the workflow"""
    max_retries: int = 3
    similarity_threshold: float = 0.3
    min_novelty_score: float = 0.7
    min_virality_score: float = 0.6
    enable_quality_control: bool = True
