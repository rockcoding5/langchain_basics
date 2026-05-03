# LangChain agent with Google Search API and calculator tools
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.callbacks import get_openai_callback
import os
import requests
from pydantic import BaseModel
from typing import List, Tuple
import numexpr

# Internal imports
from create_model.create_model import create_model
from utils.langchain_helper import get_final_answer, print_agent_token_usage


# Input schema for agent with message history
class AgentInput(BaseModel):
    messages: List[Tuple[str, str]]


# 🔢 Math calculator tool for basic arithmetic
@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations."""

    try:
        val = float(numexpr.evaluate(expression))
        return str(int(val) if val.is_integer() else val)
    except Exception as e:
        return str(e)


# 🌍 Google Search API tool for real-time web information
@tool
def search_tool(query: str) -> str:
    """Search the web for real-time information like GDP, facts, etc."""

    try:
        url = "https://www.searchapi.io/api/v1/search"

        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SEARCHAPI_API_KEY"),
        }

        response = requests.get(url, params=params)
        data = response.json()

        results = data.get("organic_results", [])

        if not results:
            return "No results found."

        snippets = [r.get("snippet", "") for r in results[:2]]
        return " ".join(snippets)

    except Exception as e:
        return str(e)


# 🤖 Agent execution with search and calculator tools
def simple_agent_google_search():
    # Create LLM instance
    llm = create_model()

    # Define available tools
    tools = [search_tool, calculator]

    # Create agent with system prompt for search and math operations
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. "
            "Use search tool to get factual data like GDP. "
            "Extract numeric values carefully from search results. "
            "Convert units into simple numeric form (e.g., 3.4 trillion → 3.4). "
            "Use calculator for math operations. "
            "Use search tool at most once unless absolutely necessary."
        )
    )

    # Execute agent with token tracking
    with get_openai_callback() as cb:
        response = agent.invoke(
            AgentInput(
                messages=[
                    ("user", "What was the GDP of India in 2022 plus 5?")
                ]
            )
        )

        # Display final answer
        print("\n🤖 Agent Response:")
        print(get_final_answer(response))

        # Show step-level token usage
        print_agent_token_usage(response)

        # Display aggregate usage metrics
        print("\n🌍 Aggregate Usage (Agent):")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")


# 🎯 Run agent example
if __name__ == "__main__":
    simple_agent_google_search()