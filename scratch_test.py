import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.content_agent import ContentAgent

async def main():
    agent = ContentAgent()
    res = await agent.generate_community_posts(
        topic="How to make a great YouTube video",
        script="Hello world, this is a great script.",
        title="10 Tips for great YT videos",
        hashtags=["#youtube", "#tips"],
        category="Education",
        channel_profile={"channel_name": "TestChannel", "tone": "Professional"}
    )
    print("Community Post Response:")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
