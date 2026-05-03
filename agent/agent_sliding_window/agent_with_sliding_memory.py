# 🤖 Agent with Tools + Sliding Memory
# Combines LLM reasoning, tool usage, and recent conversation memory

from typing import List

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from duckduckgo_search import DDGS

from pydantic import BaseModel

from create_model.create_model import create_model


# 🔧 Memory window config - number of recent messages to keep
WINDOW_SIZE = 6


# Input schema for agent with message history
class AgentInput(BaseModel):
    messages: List


# 🧮 Math calculator tool
@tool
def calculator(expression: str) -> str:
    """Math calculations"""
    import numexpr
    return str(float(numexpr.evaluate(expression)))


# 🔍 Web search tool
@tool
def search_tool(query: str) -> str:
    """Simple web search"""

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return " ".join([r["body"] for r in results])


# Get recent messages from chat history within window size
def get_window(chat_history: List):
    return chat_history[-WINDOW_SIZE:]


# 🤖 Main agent chatbot with sliding window memory
def agent_chatbot():
    # Create LLM instance
    llm = create_model()

    # Create agent with tools and system prompt
    agent = create_agent(
        model=llm,
        tools=[search_tool, calculator],
        system_prompt="You are a helpful assistant."
    )

    # Initialize chat history
    chat_history: List = []

    print("\n🤖 Chatbot Ready (type 'exit')\n")

    # Main chat loop
    while True:
        user_input = input("👤 You: ")

        if user_input.lower() == "exit":
            break

        # Process with token tracking
        with get_openai_callback() as cb:

            # Invoke agent with sliding window memory
            response = agent.invoke(
                AgentInput(
                    messages=get_window(chat_history) + [("user", user_input)]
                )
            )

            # Extract final AI response
            final_answer = None
            for msg in reversed(response["messages"]):
                if msg.type == "ai" and msg.content:
                    final_answer = msg.content
                    break

            print("\n🤖 Bot:", final_answer)

            # Store conversation in memory
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))

            print("\n🌍 Tokens:", cb.total_tokens)