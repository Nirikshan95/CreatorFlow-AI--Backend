from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import uuid
from datetime import datetime
from ..models import SessionLocal, ContentHistory, init_db
from ..agents.topic_agent import TopicAgent
from ..agents.script_agent import ScriptAgent
from ..agents.seo_agent import SEOAgent
from ..agents.content_agent import ContentAgent
from ..workflow.graph import ContentWorkflow, WorkflowConfig
from ..utils.channel_profile import channel_profile_store
from ..utils.logger import workflow_logger

router = APIRouter(prefix="/api/v1", tags=["content"])

init_db()

topic_agent = TopicAgent()
script_agent = ScriptAgent()
seo_agent = SEOAgent()
content_agent = ContentAgent()


class GenerateContentRequest(BaseModel):
    category: Optional[str] = None
    num_topics: int = 5
    script_type: Optional[str] = "descriptive"
    channel_profile_id: Optional[str] = None


class ContentResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class ChannelProfileUpdateRequest(BaseModel):
    channel_name: Optional[str] = None
    channel_link: Optional[str] = None
    script_intro_line: Optional[str] = None
    intro_line: Optional[str] = None
    description_footer: Optional[str] = None
    brand_notes: Optional[str] = None
    social_links: Optional[List[str]] = None
    useful_links: Optional[Union[List[str], List[Dict[str, str]]]] = None
    default_hashtags: Optional[List[str]] = None
    reusable_items: Optional[List[Dict[str, str]]] = None


class ChannelProfileCreateRequest(BaseModel):
    channel_name: Optional[str] = None
    channel_link: Optional[str] = None
    script_intro_line: Optional[str] = None
    intro_line: Optional[str] = None
    description_footer: Optional[str] = None
    brand_notes: Optional[str] = None
    social_links: Optional[List[str]] = None
    useful_links: Optional[Union[List[str], List[Dict[str, str]]]] = None
    default_hashtags: Optional[List[str]] = None
    reusable_items: Optional[List[Dict[str, str]]] = None


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


@router.get("/past-topics-summary", response_model=Dict)
async def get_past_topics_summary():
    db = SessionLocal()
    try:
        history = db.query(ContentHistory).order_by(ContentHistory.created_at.desc()).limit(50).all()
        topics = [h.topic for h in history]
        summary = topic_agent.build_past_topics_summary(topics)
        return {
            "summary": summary,
            "topic_count": len(topics),
        }
    finally:
        db.close()


@router.get("/channel-profile", response_model=Dict)
async def get_channel_profile():
    return channel_profile_store.load()


@router.put("/channel-profile", response_model=Dict)
async def update_channel_profile(request: ChannelProfileUpdateRequest):
    payload = request.model_dump(exclude_none=True)
    return channel_profile_store.save(payload)


@router.get("/channel-profiles", response_model=List[Dict])
async def get_channel_profiles():
    return channel_profile_store.list_profiles()


@router.post("/channel-profiles", response_model=Dict)
async def create_channel_profile(request: ChannelProfileCreateRequest):
    payload = request.model_dump(exclude_none=True)
    try:
        return channel_profile_store.create_profile(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/channel-profiles/{profile_id}", response_model=Dict)
async def update_channel_profile_by_id(profile_id: str, request: ChannelProfileUpdateRequest):
    payload = request.model_dump(exclude_none=True)
    try:
        return channel_profile_store.update_profile(profile_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel profile not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/channel-profiles/{profile_id}", response_model=Dict)
async def delete_channel_profile(profile_id: str):
    deleted = channel_profile_store.delete_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Channel profile not found")
    return {"success": True}


@router.post("/generate", response_model=ContentResponse)
async def generate_content(request: GenerateContentRequest):
    """Fallback synchronous endpoint that uses the workflow"""
    workflow = ContentWorkflow()
    generation_id = str(uuid.uuid4())
    try:
        initial_state = workflow.build_initial_state({
            "num_topics": request.num_topics,
            "category": request.category,
            "script_type": request.script_type or "descriptive",
            "channel_profile_id": request.channel_profile_id,
            "generation_id": generation_id
        })
        result = await workflow.graph.ainvoke(initial_state, config={"recursion_limit": 100})
        
        if not result.get("final_content"):
            raise HTTPException(status_code=500, detail="Generation failed")
            
        return ContentResponse(
            success=True,
            message="Content generated successfully",
            data=result["final_content"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        workflow_logger.end_generation(generation_id)


@router.get("/generate/stream")
async def generate_content_stream(
    category: Optional[str] = None,
    num_topics: int = 5,
    script_type: Optional[str] = "descriptive",
    channel_profile_id: Optional[str] = None
):
    """Real-time streaming generation using LangGraph astream"""
    
    import json
    import asyncio
    
    async def event_generator():
        workflow = ContentWorkflow()
        generation_id = str(uuid.uuid4())
        initial_state = workflow.build_initial_state({
            "num_topics": num_topics,
            "category": category,
            "script_type": script_type or "descriptive",
            "channel_profile_id": channel_profile_id,
            "generation_id": generation_id
        })
        
        workflow_logger.log_step("streaming", "start", "Connection established", generation_id=generation_id)
        
        try:
            aggregated_state = {}
            final_sent = False
            # Stream node-level updates so SSE can reflect step progress deterministically.
            stream = workflow.graph.astream(
                initial_state,
                config={"recursion_limit": 100},
                stream_mode="updates",
            )
            
            # Use an iterator directly to handle timeouts
            it = stream.__aiter__()
            
            # Start the first 'next' task
            next_task = asyncio.create_task(it.__anext__())
            
            while True:
                # Wait for either the next event OR a 15s heartbeat timeout
                done, pending = await asyncio.wait(
                    [next_task],
                    timeout=15.0
                )
                
                if next_task in done:
                    try:
                        event = next_task.result()
                        # Immediately start the next task for the next iteration
                        next_task = asyncio.create_task(it.__anext__())

                        if isinstance(event, dict):
                            # Expected in updates mode: {"node_name": {...partial state...}}
                            for node_name, state_update in event.items():
                                if isinstance(state_update, dict):
                                    aggregated_state.update(state_update)
                                yield f"data: {json.dumps({'step': node_name, 'status': 'completed'})}\n\n"
                                if node_name == "fetch_past_topics" and isinstance(state_update, dict):
                                    summary = str(state_update.get("past_topics_summary") or "").strip()
                                    if summary:
                                        yield f"data: {json.dumps({'step': 'memory_summary', 'summary': summary})}\n\n"
                                # Emit final event immediately once save_to_database provides video_id
                                if node_name == "save_to_database" and isinstance(state_update, dict):
                                    fc = state_update.get("final_content") or {}
                                    if fc.get("video_id"):
                                        yield f"data: {json.dumps({'step': 'final', 'data': fc})}\n\n"
                                        final_sent = True
                        else:
                            # Defensive fallback for non-dict stream chunks.
                            yield f"data: {json.dumps({'step': 'log', 'message': f'Unexpected stream chunk: {type(event).__name__}'})}\n\n"

                        # Flush buffered logs once per chunk.
                        for log_msg in workflow_logger.get_new_messages(generation_id):
                            yield f"data: {json.dumps({'step': 'log', 'message': log_msg})}\n\n"
                            
                    except StopAsyncIteration:
                        break
                else:
                    # Timeout reached - Send heartbeat but KEEP the next_task running
                    yield f"data: {json.dumps({'step': 'heartbeat'})}\n\n"
                    
            # Stream loop finished (workflow ended)
            final_content = aggregated_state.get("final_content")
            if final_content and not final_sent:
                # Fallback: emit final if save_to_database didn't trigger it above
                yield f"data: {json.dumps({'step': 'final', 'data': final_content})}\n\n"
            elif not final_content:
                errors = aggregated_state.get("errors", [])
                err_msg = "; ".join(errors) if errors else "Generation process aborted before completion."
                yield f"data: {json.dumps({'step': 'error', 'message': err_msg})}\n\n"
                    
        except Exception as e:
            msg = f"Stream Engine Error: {str(e)}"
            workflow_logger.log_step("workflow_engine", "error", msg, generation_id=generation_id)
            # Drain remaining logs
            for log_msg in workflow_logger.get_new_messages(generation_id):
                yield f"data: {json.dumps({'step': 'log', 'message': log_msg})}\n\n"
            yield f"data: {json.dumps({'step': 'error', 'message': msg})}\n\n"
        finally:
            workflow_logger.log_step("streaming", "end", "Connection closed", generation_id=generation_id)
            await asyncio.sleep(0.5) # Give proxy time to flush final events before TCP close
            workflow_logger.end_generation(generation_id)

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



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
    workflow = ContentWorkflow()
    generation_id = str(uuid.uuid4())
    try:
        initial_state = workflow.build_initial_state({
            "num_topics": request.num_topics,
            "category": request.category,
            "script_type": request.script_type or "descriptive",
            "channel_profile_id": request.channel_profile_id,
            "generation_id": generation_id
        })
        
        result = await workflow.graph.ainvoke(initial_state, config={"recursion_limit": 100})
        
        if not result.get("final_content"):
            raise HTTPException(
                status_code=500,
                detail=f"Content generation failed: {result.get('errors', [])}"
            )
        
        return ContentResponse(
            success=True,
            message="Content generated successfully via workflow",
            data=result["final_content"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        workflow_logger.end_generation(generation_id)
