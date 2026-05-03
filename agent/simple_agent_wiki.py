# LangChain agent with Wikipedia search and calculator tools
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.callbacks import get_openai_callback

from pydantic import BaseModel
from typing import List, Tuple
import wikipedia

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
    import numexpr
    try:
        val = float(numexpr.evaluate(expression))
        return str(int(val) if val.is_integer() else val)
    except Exception as e:
        return str(e)


# 🌍 Wikipedia search tool for factual information
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for accurate biographical facts like date of birth."""
    try:
        results = wikipedia.search(query)

        if not results:
            return "No results found."

        page = wikipedia.page(results[0])
        return page.summary[:500]

    except wikipedia.DisambiguationError as e:
        return f"Multiple results found: {e.options[:3]}"
    except Exception as e:
        return str(e)


# 🤖 Agent execution with Wikipedia and calculator tools
def simple_agent_wiki():
    # Create LLM instance
    llm = create_model()

    # Define available tools
    tools = [wikipedia_search, calculator]

    # Create agent with system prompt for Wikipedia and math operations
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. "
            "Use Wikipedia only when necessary. "
            "If results seem unrelated, stop retrying and answer using general knowledge."
        )
    )

    # Execute agent with token tracking
    with get_openai_callback() as cb:
        response = agent.invoke(
            AgentInput(
                messages=[
                    ("user", "When was Sachin Tendulkar born? What is his age in 2026?")
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
    simple_agent_wiki()