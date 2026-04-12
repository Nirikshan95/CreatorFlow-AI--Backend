from langgraph.graph import StateGraph, END
from typing import Dict, List
import uuid
import json
from ..agents.topic_agent import TopicAgent
from ..agents.script_agent import ScriptAgent
from ..agents.seo_agent import SEOAgent
from ..agents.content_agent import ContentAgent
from ..agents.critic_agent import CriticAgent
from ..models import SessionLocal, ContentHistory

from .state import ContentState, WorkflowConfig
from ..utils.vector_store import vector_store


class ContentWorkflow:
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.topic_agent = TopicAgent()
        self.script_agent = ScriptAgent()
        self.seo_agent = SEOAgent()
        self.content_agent = ContentAgent()
        self.critic_agent = CriticAgent()
        self.graph = self._build_graph()

    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ContentState)
        
        # Add nodes
        workflow.add_node("fetch_past_topics", self._fetch_past_topics)
        workflow.add_node("generate_topics", self._generate_topics)
        workflow.add_node("select_best_topic", self._select_best_topic)
        workflow.add_node("generate_script", self._generate_script)
        workflow.add_node("validate_script", self._validate_script)
        workflow.add_node("generate_seo", self._generate_seo)
        workflow.add_node("generate_content", self._generate_content)
        workflow.add_node("critique_content", self._critique_content)
        workflow.add_node("assemble_final_content", self._assemble_final_content)
        workflow.add_node("save_to_database", self._save_to_database)
        workflow.add_node("validate_topic", self._validate_topic)

        
        # Add edges
        workflow.set_entry_point("fetch_past_topics")
        workflow.add_edge("fetch_past_topics", "generate_topics")
        workflow.add_edge("generate_topics", "select_best_topic")
        workflow.add_edge("select_best_topic", "validate_topic")
        
        # Conditional edge for topic quality
        workflow.add_conditional_edges(
            "validate_topic",
            self._should_regenerate_topic,
            {
                "regenerate": "generate_topics",
                "continue": "generate_script"
            }
        )
        
        workflow.add_edge("generate_script", "validate_script")
        workflow.add_edge("validate_script", "generate_seo")
        workflow.add_edge("generate_seo", "generate_content")
        workflow.add_edge("generate_content", "critique_content")
        workflow.add_edge("critique_content", "assemble_final_content")
        workflow.add_edge("assemble_final_content", "save_to_database")

        workflow.add_edge("save_to_database", END)
        
        # Add conditional edges for quality control
        if self.config.enable_quality_control:
            workflow.add_conditional_edges(
                "validate_script",
                self._should_regenerate_script,
                {
                    "regenerate": "generate_script",
                    "continue": "generate_seo"
                }
            )
        
        return workflow.compile()
    
    def _fetch_past_topics(self, state: ContentState) -> Dict:
        """Fetch past topics from database"""
        db = SessionLocal()
        try:
            topics = db.query(ContentHistory.topic).all()
            return {"past_topics": [t[0] for t in topics]}
        except Exception as e:
            return {"errors": [f"Failed to fetch past topics: {str(e)}"], "past_topics": []}
        finally:
            db.close()

    
    def _generate_topics(self, state: ContentState) -> Dict:
        """Generate topics using TopicAgent"""
        try:
            topics = self.topic_agent.generate_topics(
                state.get("past_topics", []),
                state.get("num_topics", 5)
            )
            return {"generated_topics": topics}
        except Exception as e:
            return {"errors": [f"Failed to generate topics: {str(e)}"], "generated_topics": []}

    
    def _select_best_topic(self, state: ContentState) -> Dict:
        """Select the best topic based on scores"""
        try:
            generated = state.get("generated_topics", [])
            if not generated:
                raise ValueError("No topics generated")
            
            best_topic = self.topic_agent.select_best_topic(generated)
            return {"selected_topic": best_topic}
        except Exception as e:
            return {"errors": [f"Failed to select best topic: {str(e)}"], "selected_topic": None}


    def _validate_topic(self, state: ContentState) -> Dict:
        """Node to evaluate topic quality and update retry state"""
        selected_topic = state.get("selected_topic")
        if not selected_topic:
            return {}
            
        novelty = selected_topic.get("novelty_score", 0)
        virality = selected_topic.get("virality_score", 0)
        
        is_bad_quality = (
            novelty < self.config.min_novelty_score or 
            virality < self.config.min_virality_score
        )
        
        if is_bad_quality:
            retries = (state.get("retries") or {}).get("topic", 0)
            if retries < self.config.max_retries:
                topic_name = selected_topic.get("topic", "Unknown")
                return {
                    "retries": {"topic": retries + 1},
                    "errors": [f"Topic '{topic_name}' rejected (Novelty: {novelty:.2f}, Virality: {virality:.2f}). Regenerating... (Attempt {retries + 1})"]
                }
        
        return {}



    def _should_regenerate_topic(self, state: ContentState) -> str:
        """Pure decision function to route based on quality and retries"""
        selected_topic = state.get("selected_topic")
        if not selected_topic:
            return "continue"
            
        novelty = selected_topic.get("novelty_score", 0)
        virality = selected_topic.get("virality_score", 0)
        
        is_bad_quality = (
            novelty < self.config.min_novelty_score or 
            virality < self.config.min_virality_score
        )
        
        if is_bad_quality:
            retries = (state.get("retries") or {}).get("topic", 0)
            if retries <= self.config.max_retries and "Regenerating..." in (state["errors"][-1] if state["errors"] else ""):
                return "regenerate"
        
        return "continue"

    
    def _generate_script(self, state: ContentState) -> Dict:
        """Generate script using ScriptAgent"""
        try:
            selected = state.get("selected_topic")
            if not selected:
                raise ValueError("No topic selected")
            
            script = self.script_agent.generate_script(selected["topic"])
            return {"script": script}
        except Exception as e:
            return {"errors": [f"Failed to generate script: {str(e)}"], "script": None}

    
    def _validate_script(self, state: ContentState) -> Dict:
        """Validate the generated script and update retry state if invalid"""
        try:
            script = state.get("script")
            if not script:
                raise ValueError("No script to validate")
            
            validation = self.script_agent.validate_script(script)
            
            updates = {"script_validation": validation}
            
            if not (validation or {}).get("is_valid", False):
                retries = (state.get("retries") or {}).get("script", 0)
                if retries < self.config.max_retries:
                    updates["retries"] = {"script": retries + 1}
                    updates["errors"] = [f"Script validation failed. Regenerating... (Attempt {retries + 1})"]

            
            return updates
        except Exception as e:
            return {
                "errors": [f"Failed to validate script: {str(e)}"],
                "script_validation": {"is_valid": False, "errors": [str(e)], "warnings": []}
            }

    
    def _should_regenerate_script(self, state: ContentState) -> str:
        """Decide whether to regenerate the script based on validation state"""
        validation = state.get("script_validation")
        if not validation:
            return "continue"
        
        if not (validation or {}).get("is_valid", False):
            retries = (state.get("retries") or {}).get("script", 0)
            if retries <= self.config.max_retries and "Regenerating..." in (state["errors"][-1] if state["errors"] else ""):
                return "regenerate"

        
        return "continue"
    
    def _generate_seo(self, state: ContentState) -> Dict:
        """Generate SEO content using SEOAgent"""
        try:
            selected = state.get("selected_topic")
            if not selected:
                raise ValueError("No topic selected")
            
            # Generate titles
            titles = self.seo_agent.generate_titles(
                selected["topic"],
                selected.get("keywords", [])
            )
            
            # Select best title
            best_title = self.seo_agent.select_best_title(titles)
            
            # Generate description
            description = self.seo_agent.generate_description(
                selected["topic"],
                best_title["title"],
                selected.get("keywords", [])
            )
            
            return {
                "titles": titles,
                "selected_title": best_title,
                "description": description
            }
        except Exception as e:
            return {
                "errors": [f"Failed to generate SEO content: {str(e)}"],
                "titles": [],
                "selected_title": None,
                "description": None
            }

    
    def _generate_content(self, state: ContentState) -> Dict:
        """Generate additional content using ContentAgent"""
        try:
            selected = state.get("selected_topic")
            if not selected:
                raise ValueError("No topic selected")
            
            # Generate community posts
            community_posts = self.content_agent.generate_community_posts(
                selected["topic"]
            )
            
            # Generate thumbnail prompts
            title = (state.get("selected_title") or {}).get("title", "")
            thumbnail_prompts = self.content_agent.generate_thumbnail_prompts(
                selected["topic"],
                title
            )

            
            # Generate marketing strategy
            marketing_strategy = self.content_agent.generate_marketing_strategy(
                selected["topic"]
            )
            
            return {
                "community_posts": community_posts,
                "thumbnail_prompts": thumbnail_prompts,
                "marketing_strategy": marketing_strategy
            }
        except Exception as e:
            return {
                "errors": [f"Failed to generate content: {str(e)}"],
                "community_posts": [],
                "thumbnail_prompts": [],
                "marketing_strategy": None
            }


    def _critique_content(self, state: ContentState) -> Dict:
        """Perform final quality review using CriticAgent"""
        try:
            selected = state.get("selected_topic")
            script = state.get("script")
            if not selected or not script:
                raise ValueError("Incomplete content for critique")
            
            title = (state.get("selected_title") or {}).get("title", "")
            script_for_critique = script if isinstance(script, str) else json.dumps(script)
            critique = self.critic_agent.critique_content(
                topic=selected["topic"],
                title=title,
                script=script_for_critique,
                keywords=selected.get("keywords", [])
            )

            return {"critique": critique}
        except Exception as e:
            return {
                "errors": [f"Critique failed: {str(e)}"],
                "critique": {"rating": 5.0, "is_passed": True, "feedback": str(e)}
            }


    
    def _assemble_final_content(self, state: ContentState) -> Dict:
        """Assemble all content into final output"""
        try:
            video_id = str(uuid.uuid4())
            selected = state.get("selected_topic")
            
            final_content = {
                "video_id": video_id,
                "topic": selected["topic"] if selected else None,
                "title": (state.get("selected_title") or {}).get("title", ""),
                "script": state.get("script"),

                "script_validation": state.get("script_validation"),
                "description": state.get("description"),
                "community_post": state["community_posts"][0] if state.get("community_posts") else None,
                "thumbnail_prompt": state["thumbnail_prompts"][0] if state.get("thumbnail_prompts") else None,
                "marketing_strategy": state.get("marketing_strategy"),
                "critique": state.get("critique"),
                "all_topics": state.get("generated_topics"),
                "errors": state.get("errors", [])
            }
            return {"final_content": final_content}
        except Exception as e:
            return {
                "errors": [f"Failed to assemble final content: {str(e)}"],
                "final_content": None
            }

    
    def _save_to_database(self, state: ContentState) -> Dict:
        """Save content to database"""
        try:
            final = state.get("final_content")
            selected = state.get("selected_topic")
            if not final:
                raise ValueError("No final content to save")
            
            db = SessionLocal()
            try:
                new_content = ContentHistory(
                    id=final["video_id"],
                    video_id=final["video_id"],
                    topic=selected["topic"] if selected else "",
                    category=selected.get("category", "general") if selected else "general",
                    keywords=selected.get("keywords", []) if selected else [],
                    title=final["title"] or "",
                    script_data=final["script"],
                    seo_data={
                        "title": state.get("selected_title"),
                        "description": final["description"]
                    },
                    community_post=final["community_post"],
                    thumbnail_prompt=final["thumbnail_prompt"],
                    marketing_strategy=final["marketing_strategy"],
                    performance={"views": 0, "ctr": 0, "retention": 0},
                    novelty_score=selected.get("novelty_score", 0.5) if selected else 0.5,
                    virality_score=selected.get("virality_score", 0.5) if selected else 0.5,
                    critique_data=state.get("critique")
                )
                
                db.add(new_content)
                db.commit()

                # Add to Semantic Memory (Vector DB)
                if selected:
                    vector_store.add_topic(
                        topic_id=final["video_id"],
                        topic_text=selected["topic"],
                        metadata={
                            "category": selected.get("category", "general"),
                            "title": final["title"] or ""
                        }
                    )
                return {}
            finally:
                db.close()
        except Exception as e:
            return {"errors": [f"Failed to save to database: {str(e)}"]}

    
    def run(self, initial_state: Dict) -> ContentState:
        """Run the workflow with initial state"""
        state = {
            "past_topics": [],
            "num_topics": initial_state.get("num_topics", 5),
            "category": initial_state.get("category"),
            "generated_topics": [],
            "selected_topic": None,
            "script": None,
            "script_validation": None,
            "titles": [],
            "selected_title": None,
            "description": None,
            "community_posts": [],
            "selected_community_post": None,
            "thumbnail_prompts": [],
            "selected_thumbnail_prompt": None,
            "marketing_strategy": None,
            "final_content": None,
            "errors": [],
            "retries": {},
            "max_retries": self.config.max_retries
        }
        
        result = self.graph.invoke(state)
        return result
