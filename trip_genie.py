# --- Import all necessary libraries ---
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import ToolContext
import requests
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

# LLM_MODEL = "gemini-3.1-flash-lite-preview"
LLM_MODEL = "gemini-2.5-flash"


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


# SESSION 2

# A simple lookup to avoid needing a separate geocoding API for this example
LOCATION_COORDINATES = {
    "sunnyvale": "37.3688,-122.0363",
    "san francisco": "37.7749,-122.4194",
    "lake tahoe": "39.0968,-120.0324"
}


def get_live_weather_forecast(location: str) -> dict:
    """Gets the current, real-time weather forecast for a specified location in the US.

    Args:
        location: The city name, e.g., "San Francisco".

    Returns:
        A dictionary containing the temperature and a detailed forecast.
    """
    print(f"🛠️ TOOL CALLED: get_live_weather_forecast(location='{location}')")

    # Find coordinates for the location
    normalized_location = location.lower()
    coords_str = None
    for key, val in LOCATION_COORDINATES.items():
        if key in normalized_location:
            coords_str = val
            break
    if not coords_str:
        return {"status": "error", "message": f"I don't have coordinates for {location}."}

    try:
        # NWS API requires 2 steps: 1. Get the forecast URL from the coordinates.
        points_url = f"https://api.weather.gov/points/{coords_str}"
        headers = {"User-Agent": "ADK Example Notebook"}
        points_response = requests.get(points_url, headers=headers)
        points_response.raise_for_status()  # Raise an exception for bad status codes
        forecast_url = points_response.json()['properties']['forecast']

        # 2. Get the actual forecast from the URL.
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()

        # Extract the relevant forecast details
        current_period = forecast_response.json()['properties']['periods'][0]
        return {
            "status": "success",
            "temperature": f"{current_period['temperature']}°{current_period['temperatureUnit']}",
            "forecast": current_period['detailedForecast']
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API request failed: {e}"}

# --- Agent Definition: An agent that USES the new tool ---


weather_agent = Agent(
    name="weather_aware_planner",
    model=LLM_MODEL,
    description="A trip planner that checks the real-time weather before making suggestions.",
    instruction="You are a cautious trip planner. Before suggesting any outdoor activities, you MUST use the `get_live_weather_forecast` tool to check conditions. Incorporate the live weather details into your recommendation.",
    tools=[get_live_weather_forecast]
)

print(
    f"🌦️ Agent '{weather_agent.name}' is created and can now call a live weather API!")


async def run_weather_planner_test():
    weather_session = await session_service.create_session(app_name=weather_agent.name, user_id=my_user_id)
    query = "I want to go hiking near Lake Tahoe, what's the weather like?"
    print(f"🗣️ User Query: '{query}'")
    await run_agent_query(weather_agent, query, weather_session, my_user_id)


# Assume 'db_agent' is a pre-defined NL2SQL Agent
# For this example, we'll create placeholder agents.

db_agent = Agent(
    name="db_agent",
    model=LLM_MODEL,
    instruction="You are a database agent. When asked for data, return this mock JSON object: {'status': 'success', 'data': [{'name': 'The Grand Hotel', 'rating': 5, 'reviews': 450}, {'name': 'Seaside Inn', 'rating': 4, 'reviews': 620}]}")

# --- 1. Define the Specialist Agents ---

# The Food Critic remains the deepest specialist
food_critic_agent = Agent(
    name="food_critic_agent",
    model=LLM_MODEL,
    instruction="You are a snobby but brilliant food critic. You ONLY respond with a single, witty restaurant suggestion near the provided location.",
)

# The Concierge knows how to use the Food Critic
concierge_agent = Agent(
    name="concierge_agent",
    model=LLM_MODEL,
    instruction="You are a five-star hotel concierge. If the user asks for a restaurant recommendation, you MUST use the `food_critic_agent` tool. Present the opinion to the user politely.",
    tools=[AgentTool(agent=food_critic_agent)]
)


# --- 2. Define the Tools for the Orchestrator ---

async def call_db_agent(
    question: str,
    tool_context: ToolContext,
):
    """
    Use this tool FIRST to connect to the database and retrieve a list of places, like hotels or landmarks.
    """
    print("--- TOOL CALL: call_db_agent ---")
    agent_tool = AgentTool(agent=db_agent)
    db_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    # Store the retrieved data in the context's state
    tool_context.state["retrieved_data"] = db_agent_output
    return db_agent_output


async def call_concierge_agent(
    question: str,
    tool_context: ToolContext,
):
    """
    After getting data with call_db_agent, use this tool to get travel advice, opinions, or recommendations.
    """
    print("--- TOOL CALL: call_concierge_agent ---")
    # Retrieve the data fetched by the previous tool
    input_data = tool_context.state.get("retrieved_data", "No data found.")

    # Formulate a new prompt for the concierge, giving it the data context
    question_with_data = f"""
    Context: The database returned the following data: {input_data}

    User's Request: {question}
    """

    agent_tool = AgentTool(agent=concierge_agent)
    concierge_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    return concierge_output


# --- 3. Define the Top-Level Orchestrator Agent ---

trip_data_concierge_agent = Agent(
    name="trip_data_concierge",
    model=LLM_MODEL,
    description="Top-level agent that queries a database for travel data, then calls a concierge agent for recommendations.",
    tools=[call_db_agent, call_concierge_agent],
    instruction="""
    You are a master travel planner who uses data to make recommendations.

    1.  **ALWAYS start with the `call_db_agent` tool** to fetch a list of places (like hotels) that match the user's criteria.

    2.  After you have the data, **use the `call_concierge_agent` tool** to answer any follow-up questions for recommendations, opinions, or advice related to the data you just found.
    """,
)

print(
    f"✅ Orchestrator Agent '{trip_data_concierge_agent.name}' is defined and ready.")


# --- Let's test the Trip Data Concierge Agent ---

async def run_trip_data_concierge():
    """
    Sets up a session and runs a query against the top-level
    trip_data_concierge_agent.
    """
    # Create a new, single-use session for this query
    concierge_session = await session_service.create_session(
        app_name=trip_data_concierge_agent.name,
        user_id=my_user_id
    )

    # This query is specifically designed to trigger the full two-step process:
    # 1. Get data from the db_agent.
    # 2. Get a recommendation from the concierge_agent based on that data.
    query = "Find the top-rated hotels in San Francisco from the database, then suggest a dinner spot near the one with the most reviews."
    print(f"🗣️ User Query: '{query}'")

    # We call our existing helper function with the top-level orchestrator agent
    await run_agent_query(trip_data_concierge_agent, query, concierge_session, my_user_id)


async def main():
    # await run_day_trip_genie()
    # await run_weather_planner_test()
    await run_trip_data_concierge()


asyncio.run(main())
