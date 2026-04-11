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
from google.colab import auth
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


print("✅ All libraries are ready to go!")
