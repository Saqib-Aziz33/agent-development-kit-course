# --- Import all necessary libraries ---
import os
import sys
import json
import asyncio
import random
import string
from uuid import uuid4
from typing import Any, List

import pandas as pd
import plotly.graph_objects as go
import vertexai
from IPython.display import HTML, Markdown, display

# --- ADK, Agent, and Evaluation Components ---
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
import google.adk as adk
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.genai.types import Content, Part

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = "gemini-3.1-flash-lite-preview"


print("✅ All libraries are ready to go!")


# --- Agent Definition ---

def create_day_trip_agent():
    """Create the Spontaneous Day Trip Generator agent"""
    return Agent(
        name="day_trip_agent",
        model=LLM_MODEL,
        description="Agent specialized in generating spontaneous full-day itineraries based on mood, interests, and budget.",
        instruction="""
        You are the "Spontaneous Day Trip" Generator 🚗 - a specialized AI assistant that creates engaging full-day itineraries.

        Your Mission:
        Transform a simple mood or interest into a complete day-trip adventure with real-time details, while respecting a budget.

        Guidelines:
        1. **Budget-Aware**: Pay close attention to budget hints like 'cheap', 'affordable', or 'splurge'. Use Google Search to find activities (free museums, parks, paid attractions) that match the user's budget.
        2. **Full-Day Structure**: Create morning, afternoon, and evening activities.
        3. **Real-Time Focus**: Search for current operating hours and special events.
        4. **Mood Matching**: Align suggestions with the requested mood (adventurous, relaxing, artsy, etc.).

        RETURN itinerary in MARKDOWN FORMAT with clear time blocks and specific venue names.
        """,
        tools=[google_search]
    )


day_trip_agent = create_day_trip_agent()
print(f"🧞 Agent '{day_trip_agent.name}' is created and ready for adventure!")


# --- A Helper Function to Run Our Agents ---
# We'll use this function throughout the notebook to make running queries easy.

async def run_agent_query(agent: Agent, query: str, session: Session, user_id: str, is_router: bool = False):
    """Initializes a runner and executes a query for a given agent and session."""
    print(
        f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=agent.name
    )

    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            if not is_router:
                # Let's see what the agent is thinking!
                print(f"EVENT: {event}")
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"

    if not is_router:
        print("\n" + "-"*50)
        print("✅ Final Response:")
        display(Markdown(final_response))
        print("-"*50 + "\n")

    return final_response

# --- Initialize our Session Service ---
# This one service will manage all the different sessions in our notebook.
session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"


# --- Let's test the Day Trip Genie! ---

async def run_day_trip_genie():
    # Create a new, single-use session for this query
    day_trip_session = await session_service.create_session(
        app_name=day_trip_agent.name,
        user_id=my_user_id
    )

    # Note the new budget constraint in the query!
    query = "Plan a relaxing and artsy day trip near Sunnyvale, CA. Keep it affordable!"
    print(f"🗣️ User Query: '{query}'")

    await run_agent_query(day_trip_agent, query, day_trip_session, my_user_id)


async def main():
    await run_day_trip_genie()
