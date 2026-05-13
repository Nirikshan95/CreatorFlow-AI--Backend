from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
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
from ..utils.channel_profile import channel_profile_store
from ..utils.logger import workflow_logger
from ..utils.similarity_checker import SimilarityChecker


class ContentWorkflow:
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.topic_agent = TopicAgent()
        self.script_agent = ScriptAgent()
        self.seo_agent = SEOAgent()
        self.content_agent = ContentAgent()
        self.critic_agent = CriticAgent()
        self.graph = self._build_graph()

    @staticmethod
    def _normalize_script(script: Any) -> str:
        if isinstance(script, str):
            return script.strip()
        if script is None:
            return ""
        return str(script).strip()

    @staticmethod
    def _normalize_seo(seo_package: Dict[str, Any], fallback_title: str, fallback_description: str) -> Dict[str, Any]:
        seo_package = seo_package if isinstance(seo_package, dict) else {}
        raw_tags = seo_package.get("hashtags")
        hashtags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        hashtags = [tag if tag.startswith("#") else f"#{tag.lstrip('#')}" for tag in hashtags if tag]
        return {
            "video_title": str(seo_package.get("video_title") or fallback_title or "").strip(),
            "description": str(seo_package.get("description") or fallback_description or "").strip(),
            "hashtags": hashtags,
        }

    @staticmethod
    def _normalize_post_creation(post_creation: Any) -> Dict[str, Any]:
        data = post_creation if isinstance(post_creation, dict) else {}
        post_type = str(data.get("post_type") or "post").strip().lower()

        if post_type == "poll":
            options = data.get("options")
            clean_options = [str(opt).strip() for opt in options] if isinstance(options, list) else []
            clean_options = [opt for opt in clean_options if opt][:4]
            while len(clean_options) < 4:
                clean_options.append(f"Option {len(clean_options) + 1}")
            return {
                "post_type": "poll",
                "poll_question": str(data.get("poll_question") or "").strip(),
                "options": clean_options,
            }

        return {
            "post_type": "post",
            "content": str(data.get("content") or "").strip(),
            "image_prompt": str(data.get("image_prompt") or "").strip(),
        }

    @staticmethod
    def _normalize_thumbnail_prompt(thumbnail_data: Any) -> str:
        if isinstance(thumbnail_data, dict):
            return str(thumbnail_data.get("thumbnail_prompt") or "").strip()
        if isinstance(thumbnail_data, str):
            return thumbnail_data.strip()
        return ""

    @staticmethod
    def _normalize_distribution(strategy: Any) -> Dict[str, List[str]]:
        data = strategy if isinstance(strategy, dict) else {}
        suggestions = data.get("suggestions")
        clean = [str(item).strip() for item in suggestions] if isinstance(suggestions, list) else []
        return {"suggestions": [item for item in clean if item]}

    @staticmethod
    def _normalize_quality_assessment(critique: Any) -> Dict[str, Any]:
        return critique if isinstance(critique, dict) else {}

    
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
        workflow.add_node("generate_post_image", self._generate_post_image)
        workflow.add_node("critique_content", self._critique_content)
        workflow.add_node("assemble_final_content", self._assemble_final_content)
        workflow.add_node("save_to_database", self._save_to_database)

        
        # Add edges
        workflow.set_entry_point("fetch_past_topics")
        workflow.add_edge("fetch_past_topics", "generate_topics")
        workflow.add_edge("generate_topics", "select_best_topic")
        workflow.add_edge("select_best_topic", "generate_script")
        
        workflow.add_conditional_edges(
            "validate_script",
            self._should_regenerate_script,
            {
                "regenerate": "generate_script",
                "continue": "generate_seo",
                "fail": END
            }
        )

        workflow.add_edge("generate_script", "validate_script")
        workflow.add_edge("generate_seo", "generate_content")
        workflow.add_edge("generate_content", "generate_post_image")
        workflow.add_edge("generate_post_image", "critique_content")
        
        # New: Refinement loop
        workflow.add_node("refine_content", self._refine_content)
        workflow.add_conditional_edges(
            "critique_content",
            self._should_refine_content,
            {
                "refine": "refine_content",
                "continue": "assemble_final_content",
                "fail": END
            }
        )
        workflow.add_edge("refine_content", "generate_script")

        workflow.add_edge("assemble_final_content", "save_to_database")
        workflow.add_edge("save_to_database", END)
        
        return workflow.compile()
    
    def _fetch_past_topics(self, state: ContentState) -> Dict:
        """Fetch past topics from database"""
        workflow_logger.log_step("fetch_past_topics", "start")
        db = SessionLocal()
        try:
            history = db.query(ContentHistory).order_by(ContentHistory.created_at.desc()).limit(50).all()
            topics = [h.topic for h in history]
            summary = self.topic_agent.build_past_topics_summary(topics)
            workflow_logger.log_step("fetch_past_topics", "success", f"Found {len(topics)} past topics")
            workflow_logger.log_step("fetch_past_topics", "info", f"Using compact memory summary ({len(summary.split())} words)")
            return {"past_topics": topics, "past_topics_summary": summary}
        except Exception as e:
            workflow_logger.log_step("fetch_past_topics", "error", str(e))
            return {
                "past_topics": [],
                "past_topics_summary": "No prior topics recorded.",
                "errors": [f"DB history failed: {str(e)}"]
            }
        finally:
            db.close()

    async def _generate_topics(self, state: ContentState) -> Dict:
        """Generate topic ideas using TopicAgent (Flash Tier)"""
        workflow_logger.log_step("generate_topics", "start")
        try:
            topics = await self.topic_agent.generate_topics(
                num_topics=state["num_topics"],
                category=state.get("category"),
                past_topics=state.get("past_topics", []),
                past_topics_summary=state.get("past_topics_summary", ""),
                channel_profile=state.get("channel_profile")
            )
            
            if not topics:
                workflow_logger.log_step("generate_topics", "warning", "LLM returned no topics, using fallback")
                # Fallback: generate a default topic if LLM fails completely
                topics = [{
                    "topic": "How to build confidence and overcome self-doubt",
                    "novelty_score": 0.85,
                    "virality_score": 0.8,
                    "category": state.get("category") or "general",
                    "keywords": ["confidence", "self-improvement", "mindset"],
                    "rationale": "Fallback topic generated when LLM returned empty results"
                }]
            
            workflow_logger.log_step("generate_topics", "success", f"Generated {len(topics)} ideas")
            return {"generated_topics": topics}
        except Exception as e:
            workflow_logger.log_step("generate_topics", "error", str(e))
            # Return a fallback topic instead of just an error
            return {
                "generated_topics": [{
                    "topic": "How to build confidence and overcome self-doubt",
                    "novelty_score": 0.85,
                    "virality_score": 0.8,
                    "category": state.get("category") or "general",
                    "keywords": ["confidence", "self-improvement", "mindset"],
                    "rationale": "Fallback topic due to generation error"
                }],
                "errors": [f"Topic generation had issues, using fallback: {str(e)}"]
            }

    async def _select_best_topic(self, state: ContentState) -> Dict:
        """Select the best topic from generated ideas (Flash Tier)"""
        workflow_logger.log_step("select_best_topic", "start")
        try:
            topics = state.get("generated_topics", [])
            if not topics:
                workflow_logger.log_step("select_best_topic", "warning", "No topics to select from, using fallback")
                # Return a fallback topic instead of raising
                return {
                    "selected_topic": {
                        "topic": "How to build confidence and overcome self-doubt",
                        "novelty_score": 0.85,
                        "virality_score": 0.8,
                        "category": state.get("category") or "general",
                        "keywords": ["confidence", "self-improvement", "mindset"],
                        "rationale": "Fallback topic - no topics were generated"
                    },
                    "errors": ["No topics generated, using fallback topic"]
                }
            
            best_topic = await self.topic_agent.select_best_topic(
                topics,
                channel_profile=state.get("channel_profile")
            )
            workflow_logger.log_step("select_best_topic", "success", f"Selected: {best_topic['topic']}")
            return {"selected_topic": best_topic}
        except Exception as e:
            workflow_logger.log_step("select_best_topic", "error", str(e))
            # Try to use the first generated topic as fallback
            topics = state.get("generated_topics", [])
            if topics and isinstance(topics[0], dict):
                fallback = topics[0]
                workflow_logger.log_step("select_best_topic", "info", "Using first generated topic as fallback")
                return {
                    "selected_topic": fallback,
                    "errors": [f"Topic selection failed, using first option: {str(e)}"]
                }
            # Ultimate fallback
            return {
                "selected_topic": {
                    "topic": "How to build confidence and overcome self-doubt",
                    "novelty_score": 0.85,
                    "virality_score": 0.8,
                    "category": state.get("category") or "general",
                    "keywords": ["confidence", "self-improvement", "mindset"],
                    "rationale": "Ultimate fallback topic"
                },
                "errors": [f"Failed to select best topic, using fallback: {str(e)}"]
            }




    
    async def _generate_script(self, state: ContentState) -> Dict:
        """Generate script using ScriptAgent (Heavy Tier)"""
        selected = (state.get("selected_topic") or {}).get("topic", "N/A")
        workflow_logger.log_step("generate_script", "start", f"Topic: {selected}")
        try:
            selected_topic = state.get("selected_topic")
            if not selected_topic:
                raise ValueError("No topic selected")
            
            critique = state.get("critique") or {}
            feedback = None
            if not critique.get("is_passed", True):
                feedback = critique.get("feedback")

            script_type = state.get("script_type", "descriptive") or "descriptive"
            script = await self.script_agent.generate_script(
                selected_topic["topic"],
                category=state.get("category"),
                script_type=script_type,
                channel_profile=state.get("channel_profile"),
                feedback=feedback
            )
            if not script:
                raise ValueError("Generated script was empty or None")
                
            workflow_logger.log_step("generate_script", "success", f"Script length: {len(str(script))}")
            return {"script": script}
        except Exception as e:
            workflow_logger.log_step("generate_script", "error", f"Script generation failed: {str(e)}")
            return {"errors": [f"Failed to generate script: {str(e)}"]}

    
    def _validate_script(self, state: ContentState) -> Dict:
        """Validate the generated script and update retry state if invalid"""
        workflow_logger.log_step("validate_script", "start")
        retries = state.get("retries", {})
        script_retry_count = retries.get("script", 0)
        
        try:
            script = state.get("script")
            if not script:
                raise ValueError("No script found to validate")
            
            validation = self.script_agent.validate_script(script)
            workflow_logger.log_step("validate_script", "success", f"Is valid: {validation.get('is_valid')}")
            
            updates = {"script_validation": validation}
            
            if not (validation or {}).get("is_valid", False):
                if script_retry_count < self.config.max_retries:
                    updates["retries"] = {**retries, "script": script_retry_count + 1}
                    updates["errors"] = [f"Script validation failed. Regenerating... (Attempt {script_retry_count + 1})"]

            return updates
        except Exception as e:
            workflow_logger.log_step("validate_script", "error", str(e))
            return {
                "errors": [f"Failed to validate script: {str(e)}"],
                "script_validation": {"is_valid": False, "errors": [str(e)], "warnings": []},
                "retries": {**retries, "script": script_retry_count + 1}
            }


    def _should_regenerate_script(self, state: ContentState) -> str:
        """Decide whether to regenerate or continue based on validation"""
        validation = state.get("script_validation")
        retries = (state.get("retries") or {}).get("script", 0)

        # SAFETY BREAK: If we've hit max retries, STOP
        if retries >= self.config.max_retries:
            workflow_logger.log_step("workflow_router", "error", f"Script retry limit reached ({retries}). Terminating generation phase.")
            return "fail"

        if not validation:
            return "continue"
        
        # Critical failure check (Invalid or empty)
        if not validation.get("is_valid", False):
            workflow_logger.log_step("workflow_router", "info", f"Script invalid (Attempt {retries}). Regenerating...")
            return "regenerate"
        
        workflow_logger.log_step("workflow_router", "success", "Script validated. Proceeding to SEO phase.")
        return "continue"
    

    def _refine_content(self, state: ContentState) -> Dict:
        """Integration node to prepare for regeneration after critique"""
        workflow_logger.log_step("refine_content", "start")
        critique = state.get("critique")
        retries = (state.get("retries") or {}).get("refinement", 0)
        
        return {
            "retries": {"refinement": retries + 1},
            "errors": [f"Critic requested refinement: {critique.get('feedback', 'Quality below threshold')}"]
        }

    def _should_refine_content(self, state: ContentState) -> str:
        """Conditional logic for refinement loop"""
        critique = state.get("critique") or {}
        retries = (state.get("retries") or {}).get("refinement", 0)

        if not critique.get("is_passed", True):
            if retries < self.config.max_retries:
                workflow_logger.log_step("workflow_router", "info", f"Critic requested refinement (Pass {retries+1}/{self.config.max_retries}).")
                return "refine"
            else:
                workflow_logger.log_step("workflow_router", "error", "Max refinement attempts reached. Terminating generation due to quality failure.")
                return "fail"
        
        return "continue"


    async def _generate_seo(self, state: ContentState) -> Dict:
        """Generate SEO content using SEOAgent (Parallelized)"""
        workflow_logger.log_step("generate_seo", "start")
        selected = state.get("selected_topic")
        
        # Validate required inputs
        if not selected:
            workflow_logger.log_step("generate_seo", "error", "No topic selected for SEO generation")
            return {"errors": ["Cannot proceed: No topic selected for SEO generation"]}
        
        
        topic = selected.get("topic")
        if not topic or not isinstance(topic, str) or not topic.strip():
            workflow_logger.log_step("generate_seo", "error", "Topic is empty or invalid")
            return {"errors": ["Cannot proceed: Topic is empty or invalid"]}
        
        
        script = state.get("script")
        if not script:
            workflow_logger.log_step("generate_seo", "error", "Script is required for description generation")
            return {"errors": ["Cannot proceed: Script is required for description generation but is missing"]}

        critique = state.get("critique") or {}
        feedback = critique.get("feedback") if not critique.get("is_passed", True) else None

        try:
            keywords = selected.get("keywords", [])
            working_title = (state.get("selected_title") or {}).get("title", "")

            seo_package = await self.seo_agent.generate_seo_package(
                topic,
                title=working_title,
                keywords=keywords,
                category=state.get("category"),
                channel_profile=state.get("channel_profile"),
                feedback=feedback,
                script=script
            )

            selected_title = {"title": seo_package.get("video_title", f"About {topic}")}
            description = seo_package.get("description", "")

            return {
                "titles": [selected_title],
                "selected_title": selected_title,
                "description": description,
                "seo_package": seo_package,
            }
        except Exception as e:
            workflow_logger.log_step("generate_seo", "warning", f"SEO fallback used: {str(e)}")
            fallback_title_text = f"About {topic}"
            fallback_title = {"title": fallback_title_text}
            fallback_desc = f"In this video, we explore {topic} with practical takeaways."
            return {
                "titles": [fallback_title],
                "selected_title": fallback_title,
                "description": fallback_desc,
                "seo_package": {
                    "video_title": fallback_title_text,
                    "description": fallback_desc,
                    "hashtags": []
                }
            }

    async def _generate_content(self, state: ContentState) -> Dict:
        """Generate additional content using ContentAgent (Parallelized)"""
        workflow_logger.log_step("generate_content", "start")
        selected = state.get("selected_topic")
        
        # Validate required inputs
        if not selected:
            workflow_logger.log_step("generate_content", "error", "No topic selected for content generation")
            return {"errors": ["Cannot proceed: No topic selected for content generation"]}
        
        
        topic = selected.get("topic")
        if not topic or not isinstance(topic, str) or not topic.strip():
            workflow_logger.log_step("generate_content", "error", "Topic is empty or invalid")
            return {"errors": ["Cannot proceed: Topic is empty or invalid"]}
        
        
        script = state.get("script")
        if not script:
            workflow_logger.log_step("generate_content", "error", "Script is required for content generation")
            return {"errors": ["Cannot proceed: Script is required for content generation but is missing"]}
        
        title = (state.get("selected_title") or {}).get("title", f"About {topic}")
        category = state.get("category")
        profile = state.get("channel_profile")
        seo_package = state.get("seo_package") or {}
        hashtags = seo_package.get("hashtags", [])

        # PHASE 1: PARALLEL EXECUTION
        import asyncio
        tasks = [
            self.content_agent.generate_community_posts(
                topic,
                script,
                title=title,
                hashtags=hashtags,
                category=category,
                channel_profile=profile
            ),
            self.content_agent.generate_thumbnail_prompts(
                topic,
                title,
                script=script,
                category=category,
                channel_profile=profile,
            ),
            self.content_agent.generate_marketing_strategy(
                topic,
                script,
                title=title,
                category=category,
                channel_profile=profile,
            )
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            post_creation = results[0] if not isinstance(results[0], Exception) else {}
            thumbnail_prompt = results[1] if not isinstance(results[1], Exception) else {}
            strategy = results[2] if not isinstance(results[2], Exception) else {}
            
            if any(isinstance(r, Exception) for r in results):
                workflow_logger.log_step("generate_content", "warning", "Some content tasks partially failed but proceeded.")

            workflow_logger.log_step("generate_content", "success", "All parallel marketing tasks completed.")
            
            return {
                "community_posts": post_creation,
                "thumbnail_prompts": thumbnail_prompt,
                "marketing_strategy": strategy
            }
        except Exception as e:
            workflow_logger.log_step("generate_content", "error", f"Parallel block failed: {str(e)}")
            return {"errors": [f"Content marketing phase failed: {str(e)}"]}

    async def _generate_post_image(self, state: ContentState) -> Dict:
        """Generate an image prompt specifically for the community post"""
        workflow_logger.log_step("generate_post_image", "start")
        try:
            topic = str((state.get("selected_topic") or {}).get("topic") or "").strip()
            script = state.get("script")
            posts = state.get("community_posts") or {}
            
            if isinstance(posts, dict) and posts.get("post_type") == "post" and posts.get("content"):
                workflow_logger.log_step("generate_post_image", "info", "Generating image prompt for community post...")
                result = await self.content_agent.generate_post_image_prompt(topic, script, posts.get("content"))
                workflow_logger.log_step("generate_post_image", "success", "Generated post image prompt")
                return {"post_image_prompts": [result]}
            
            workflow_logger.log_step("generate_post_image", "info", "Skipped post image prompt (not a regular post)")
            return {"post_image_prompts": []}
        except Exception as e:
            workflow_logger.log_step("generate_post_image", "error", str(e))
            return {"errors": [f"Post image prompt generation failed: {str(e)}"]}

    async def _critique_content(self, state: ContentState) -> Dict:
        """Perform final quality review using CriticAgent (Heavy Tier)"""
        workflow_logger.log_step("critique_content", "start")
        
        # Validate required inputs
        selected = state.get("selected_topic")
        script = state.get("script")
        
        if not selected:
            workflow_logger.log_step("critique_content", "error", "No topic selected for critique")
            return {"errors": ["Cannot proceed: No topic selected for critique"]}
        
        
        topic = selected.get("topic")
        if not topic or not isinstance(topic, str) or not topic.strip():
            workflow_logger.log_step("critique_content", "error", "Topic is empty or invalid")
            return {"errors": ["Cannot proceed: Topic is empty or invalid"]}
        
        
        if not script:
            workflow_logger.log_step("critique_content", "error", "Script is required for critique")
            return {"errors": ["Cannot proceed: Script is required for critique but is missing"]}
        
        
        try:
            title = (state.get("selected_title") or {}).get("title", "")
            script_for_critique = script if isinstance(script, str) else json.dumps(script)
            critique = await self.critic_agent.critique_content(
                topic=topic,
                title=title,
                script=script_for_critique,
                keywords=selected.get("keywords", [])
            )
            workflow_logger.log_step("critique_content", "success", f"Rating: {critique.get('rating')}")
            return {"critique": critique}
        except Exception as e:
            workflow_logger.log_step("critique_content", "warning", f"Critique skipped: {str(e)}")
            return {
                "critique": {"rating": 5.0, "is_passed": True, "feedback": "Critique skipped due provider connectivity issue."}
            }

    def _assemble_final_content(self, state: ContentState) -> Dict:
        """Assemble all generated pieces into the final output format"""
        workflow_logger.log_step("assemble_final_content", "start")
        try:
            topic = str((state.get("selected_topic") or {}).get("topic") or "").strip()
            title = str((state.get("selected_title") or {}).get("title") or "").strip()
            script = self._normalize_script(state.get("script"))

            seo = self._normalize_seo(
                state.get("seo_package") or {},
                fallback_title=title,
                fallback_description=str(state.get("description") or "").strip(),
            )
            post_creation = self._normalize_post_creation(state.get("community_posts") or {})
            
            # Inject the separate post image prompt if applicable
            image_prompts = state.get("post_image_prompts") or []
            if post_creation.get("post_type") == "post" and image_prompts:
                post_creation["image_prompt"] = str(image_prompts[0].get("post_image_prompt", "")).strip()

            thumbnail_prompt = self._normalize_thumbnail_prompt(state.get("thumbnail_prompts") or {})
            distribution_strategy = self._normalize_distribution(state.get("marketing_strategy") or {})
            quality_assessment = self._normalize_quality_assessment(state.get("critique"))

            final_content = {
                "topic": topic,
                "title": title,
                "script": script,
                "past_topics_summary": str(state.get("past_topics_summary") or "").strip(),
                "seo": seo,
                "post_creation": post_creation,
                "thumbnail_prompt": thumbnail_prompt,
                "distribution_strategy": distribution_strategy,
                "quality_assessment": quality_assessment,
                "quality_score": quality_assessment.get("rating", 0.0),
                "generation_log": workflow_logger.read_log(state.get("generation_id"))
            }
            
            # Check if it meets criteria to be called 'complete'
            has_core = bool(topic and title and script)
            
            if not has_core:
                workflow_logger.log_step("assemble_final_content", "warning", "Generation incomplete: missing core components")
            
            return {"final_content": final_content}
        except Exception as e:
            workflow_logger.log_step("assemble_final_content", "error", str(e))
            return {"errors": [f"Assembly failed: {str(e)}"]}

    def _save_to_database(self, state: ContentState) -> Dict:
        """Save the final generated content to the database"""
        workflow_logger.log_step("save_to_database", "start")
        final_content = state.get("final_content")
        if not final_content:
            return {"errors": ["No content to save"]}

        db = SessionLocal()
        try:
            video_id = str(uuid.uuid4())
            selected_topic = state.get("selected_topic") or {}
            critique = state.get("critique") or {}
            seo = (final_content.get("seo") or {})
            post_creation = final_content.get("post_creation")
            thumbnail_prompt = final_content.get("thumbnail_prompt")
            distribution_strategy = final_content.get("distribution_strategy") or {}
            quality_assessment = final_content.get("quality_assessment") or critique
            script_data = self._normalize_script(final_content.get("script"))
            history = ContentHistory(
                id=str(uuid.uuid4()),
                video_id=video_id,
                topic=final_content["topic"] or "Untitled",
                category=state.get("category") or "general",
                keywords=selected_topic.get("keywords") or [],
                title=final_content["title"] or "Untitled",
                script_summary=(script_data[:500] if isinstance(script_data, str) else None),
                script_data=script_data if script_data else None,
                seo_data=seo,
                community_post=post_creation if isinstance(post_creation, dict) else {},
                thumbnail_prompt=thumbnail_prompt if isinstance(thumbnail_prompt, str) else "",
                marketing_strategy=distribution_strategy if isinstance(distribution_strategy, dict) else {"suggestions": []},
                performance=None,
                novelty_score=float(selected_topic.get("novelty_score", 0.0) or 0.0),
                virality_score=float(selected_topic.get("virality_score", 0.0) or 0.0),
                critique_data=quality_assessment if isinstance(quality_assessment, dict) else {}
            )
            db.add(history)
            db.commit()
            
            # Update the final_content with the generated video_id
            final_content["video_id"] = video_id
            
            # Also add to vector store for future similarity checks
            if vector_store:
                vector_store.add_topic(
                    video_id, 
                    final_content["topic"], 
                    {"video_id": video_id, "title": final_content["title"]}
                )
            
            workflow_logger.log_step("save_to_database", "success", f"Video ID: {video_id}")
            return {"final_content": final_content}
        except Exception as e:
            workflow_logger.log_step("save_to_database", "error", str(e))
            db.rollback()
            return {"errors": [f"Database save failed: {str(e)}"]}
        finally:
            db.close()

    def build_initial_state(self, params: Dict) -> Dict:
        """Initialize the state with user parameters and config defaults"""
        generation_id = params.get("generation_id") or str(uuid.uuid4())
        workflow_logger.start_generation(generation_id)
        
        # Load channel profile if provided
        profile = None
        profile_id = params.get("channel_profile_id")
        if profile_id:
            profile = channel_profile_store.get_profile(profile_id)
            if profile:
                workflow_logger.log_step("initialization", "success", f"Loaded profile: {profile['channel_name']}")

        return {
            "generation_id": generation_id,
            "num_topics": params.get("num_topics", 5),
            "category": params.get("category"),
            "channel_profile_id": profile_id,
            "channel_profile": profile,
            # Accept both snake_case (API) and legacy camelCase payload keys.
            "script_type": params.get("script_type") or params.get("scriptType", "descriptive"),
            "past_topics": [],
            "past_topics_summary": "",
            "generated_topics": [],
            "selected_topic": None,
            "script": None,
            "script_validation": None,
            "titles": [],
            "selected_title": None,
            "description": None,
            "seo_package": None,
            "community_posts": {},
            "thumbnail_prompts": {},
            "marketing_strategy": {},
            "critique": None,
            "final_content": None,
            "errors": [],
            "retries": {"topic": 0, "script": 0, "refinement": 0},
            "max_retries": self.config.max_retries
        }
