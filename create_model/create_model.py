# 🧠 Centralized LLM Factory
# Ensures consistent model configuration across all project components
# (chains, agents, memory, UI)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables including OPENAI_API_KEY
load_dotenv()


def create_model():
    """
    Returns a configured ChatOpenAI instance.

    Configuration choices:
    - temperature=0.3 → stable, predictable outputs for learning/debugging
    - max_tokens=200 → prevents excessive response length
    - max_retries=2 → resilience for API reliability issues
    """

    return ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.3,
        max_tokens=200,
        max_retries=2,
    )