from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
from datetime import datetime
from ..models import SessionLocal, ContentHistory, init_db
from ..agents.topic_agent import TopicAgent
from ..agents.script_agent import ScriptAgent
from ..agents.seo_agent import SEOAgent
from ..agents.content_agent import ContentAgent
from ..workflow.graph import ContentWorkflow, WorkflowConfig

router = APIRouter(prefix="/api/v1", tags=["content"])

init_db()

topic_agent = TopicAgent()
script_agent = ScriptAgent()
seo_agent = SEOAgent()
content_agent = ContentAgent()
workflow = ContentWorkflow()


class GenerateContentRequest(BaseModel):
    category: Optional[str] = None
    num_topics: int = 5


class ContentResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


@router.get("/history", response_model=List[Dict])
async def get_content_history():
    db = SessionLocal()
    try:
        history = db.query(ContentHistory).order_by(ContentHistory.created_at.desc()).limit(50).all()
        return [
            {
                "video_id": h.video_id,
                "topic": h.topic,
                "category": h.category,
                "title": h.title,
                "novelty_score": h.novelty_score,
                "virality_score": h.virality_score,
                "created_at": h.created_at.isoformat()
            }
            for h in history
        ]
    finally:
        db.close()


@router.get("/past-topics", response_model=List[str])
async def get_past_topics():
    db = SessionLocal()
    try:
        topics = db.query(ContentHistory.topic).all()
        return [t[0] for t in topics]
    finally:
        db.close()


@router.post("/generate", response_model=ContentResponse)
async def generate_content(request: GenerateContentRequest):
    """Fallback synchronous endpoint that uses the workflow"""
    try:
        initial_state = {
            "num_topics": request.num_topics,
            "category": request.category
        }
        result = workflow.run(initial_state)
        
        if not result["final_content"]:
            raise HTTPException(status_code=500, detail="Generation failed")
            
        return ContentResponse(
            success=True,
            message="Content generated successfully",
            data=result["final_content"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/stream")
async def generate_content_stream(category: Optional[str] = None, num_topics: int = 5):
    """Real-time streaming generation using LangGraph astream"""
    
    async def event_generator():
        initial_state = {
            "num_topics": num_topics,
            "category": category,
            "past_topics": [],
            "generated_topics": [],
            "errors": [],
            "retries": {},
            "max_retries": 3
        }
        
        try:
            aggregated_state = {}
            # We use the compiled graph directly for streaming
            async for event in workflow.graph.astream(initial_state):
                # LangGraph events look like { 'node_name': { 'state_update' } }
                for node_name, state_update in event.items():
                    if isinstance(state_update, dict):
                        aggregated_state.update(state_update)

                    # Send node completion status
                    yield f"data: {json.dumps({'step': node_name, 'status': 'completed'})}\n\n"
                    
                    # If this is the final storage node, extract and send the final data
                    if node_name == "save_to_database":
                        final_content = aggregated_state.get("final_content")
                        yield f"data: {json.dumps({'step': 'final', 'data': final_content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    import json
    return StreamingResponse(event_generator(), media_type="text/event-stream")



@router.get("/content/{video_id}", response_model=Dict)
async def get_content_by_id(video_id: str):
    db = SessionLocal()
    try:
        content = db.query(ContentHistory).filter(ContentHistory.video_id == video_id).first()
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "video_id": content.video_id,
            "topic": content.topic,
            "category": content.category,
            "keywords": content.keywords,
            "title": content.title,
            "script_data": content.script_data,
            "seo_data": content.seo_data,
            "community_post": content.community_post,
            "thumbnail_prompt": content.thumbnail_prompt,
            "marketing_strategy": content.marketing_strategy,
            "performance": content.performance,
            "novelty_score": content.novelty_score,
            "virality_score": content.virality_score,
            "critique": content.critique_data,
            "created_at": content.created_at.isoformat()

        }
    finally:
        db.close()


@router.post("/generate/workflow", response_model=ContentResponse)
async def generate_content_workflow(request: GenerateContentRequest):
    try:
        initial_state = {
            "num_topics": request.num_topics,
            "category": request.category
        }
        
        result = workflow.run(initial_state)
        
        if not result["final_content"]:
            raise HTTPException(
                status_code=500,
                detail=f"Content generation failed: {result['errors']}"
            )
        
        return ContentResponse(
            success=True,
            message="Content generated successfully via workflow",
            data=result["final_content"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
