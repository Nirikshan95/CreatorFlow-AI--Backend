from typing import TypedDict, List, Dict, Optional, Annotated, Any
import operator
from pydantic import BaseModel


def merge_dict(a: Dict, b: Dict) -> Dict:
    """Simple dictionary merge reducer"""
    return {**a, **b}


class ContentState(TypedDict):
    """State for the content generation workflow"""
    past_topics: Annotated[List[str], operator.add]
    past_topics_summary: str
    generation_id: str

    num_topics: int
    category: Optional[str]
    channel_profile_id: Optional[str]
    channel_profile: Optional[Dict]
    
    # Topic generation
    generated_topics: List[Dict]
    selected_topic: Optional[Dict]
    
    # Script generation
    script_type: Optional[str]
    script: Optional[Any]
    script_validation: Optional[Dict]
    
    # SEO generation
    titles: List[Dict]
    selected_title: Optional[Dict]
    description: Optional[Any]
    seo_package: Optional[Dict]
    
    # Content generation
    community_posts: Optional[Any]
    selected_community_post: Optional[Dict]
    thumbnail_prompts: Optional[Any]
    selected_thumbnail_prompt: Optional[Dict]
    post_image_prompts: List[Dict]
    selected_post_image_prompt: Optional[Dict]
    marketing_strategy: Optional[Dict]
    critique: Optional[Dict]
    
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
