# 🧠 Sliding Window Memory
# Keeps only recent conversation to control token usage

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model


# Window size configuration - last N messages (not full history)
WINDOW_SIZE = 6  # = last 3 turns


def get_window(chat_history: List):
    """
    Returns only recent messages from chat history

    Benefits:
    - Prevents token explosion
    - Keeps context relevant
    """
    return chat_history[-WINDOW_SIZE:]


def sliding_window_chain():
    # Create LLM instance
    llm = create_model()
    # Initialize chat history
    chat_history: List = []

    # Create chat prompt template with history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # Build chain with prompt and LLM
    chain = prompt | llm

    # Execute with token tracking
    with get_openai_callback() as cb:

        user_input = "Suggest a fancy restaurant name for Indian food"

        response = chain.invoke({
            "input": user_input,
            "chat_history": get_window(chat_history)
        })

        print("\n🏷️ Response:", response.content)

        # Store conversation in memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

        print("\n🌍 Tokens:", cb.total_tokens)