# Redis-based memory chain for persistent conversation history
import json
import redis
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model


# Redis connection and session configuration
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
SESSION_ID = "user:demo"


# Load conversation history from Redis
def load_history() -> List:
    items = r.lrange(SESSION_ID, 0, -1)
    messages = []
    for item in items:
        m = json.loads(item)
        if m["type"] == "human":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    return messages


# Save new message to Redis
def append_message(msg):
    payload = json.dumps({"type": msg.type, "content": msg.content})
    r.rpush(SESSION_ID, payload)


def redis_memory_chain():
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

        user_input = "Suggest a menu for the above restaurant"
        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        print("\n🍽️ Response:", response.content)

        # Save conversation to Redis memory
        append_message(HumanMessage(content=user_input))
        append_message(AIMessage(content=response.content))

        print("\n🌍 Tokens:", cb.total_tokens)


# Run Redis memory chain example
if __name__ == "__main__":
    redis_memory_chain()
