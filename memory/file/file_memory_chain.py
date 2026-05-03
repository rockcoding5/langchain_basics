# File-based memory chain for persistent conversation history
import json
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model


# File path for storing chat history
FILE_PATH = Path("chat_history.json")


# Load conversation history from file
def load_history() -> List:
    if FILE_PATH.exists():
        data = json.loads(FILE_PATH.read_text())
        messages = []
        for m in data:
            if m["type"] == "human":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
        return messages
    return []


# Save conversation history to file
def save_history(chat_history):
    data = [{"type": m.type, "content": m.content} for m in chat_history]
    FILE_PATH.write_text(json.dumps(data, indent=2))


def file_memory_chain():
    # Create LLM instance
    llm = create_model()
    # Load existing conversation history
    chat_history = load_history()

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
            "chat_history": chat_history
        })

        print("\n🏷️ Response:", response.content)

        # Save conversation to file memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))
        save_history(chat_history)

        print("\n🌍 Tokens:", cb.total_tokens)


# Run file memory chain example
if __name__ == "__main__":
    file_memory_chain()
