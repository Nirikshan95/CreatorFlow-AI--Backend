import sys
from pathlib import Path
import asyncio

# Add backend to path properly
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

from app.workflow.graph import ContentWorkflow
from app.workflow.state import WorkflowConfig

async def test_workflow():
    print("Initializing workflow...")
    config = WorkflowConfig(max_retries=1)
    workflow = ContentWorkflow(config=config)
    
    initial_state = {
        "num_topics": 2,
        "category": "productivity",
        "past_topics": [],
        "generated_topics": [],
        "errors": [],
        "retries": {},
        "max_retries": 1
    }
    
    print("Running workflow...")
    try:
        # We'll use invoke for simplicity in this test
        result = workflow.graph.invoke(initial_state)
        
        print("\n--- Workflow Result ---")
        if result.get("final_content"):
            print("SUCCESS: Content generated!")
            print(f"Topic: {result['final_content'].get('topic')}")
        else:
            print("FAILED: No final content.")
            
        if result.get("errors"):
            print("\nErrors encountered:")
            for err in result["errors"]:
                print(f"- {err}")
                
        print(f"\nRetries: {result.get('retries')}")
        
    except Exception as e:
        print(f"Workflow crashed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_workflow())
