# Vector database memory chain for semantic search and retrieval
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from create_model.create_model import create_model


# Initialize Chroma vector store with OpenAI embeddings
vectorstore = Chroma(
    collection_name="chat_memory",
    embedding_function=OpenAIEmbeddings()
)


# Save text to vector database
def save_to_memory(text: str):
    vectorstore.add_texts([text])


# Retrieve relevant memories using semantic search
def retrieve_memory(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


def vector_memory_chain():
    # Create LLM instance
    llm = create_model()

    # Create prompt template with context integration
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context below if relevant."),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    # Build chain with prompt and LLM
    chain = prompt | llm

    # Execute with token tracking
    with get_openai_callback() as cb:

        user_input = "Suggest a fancy restaurant name for Indian food"

        # Retrieve relevant context from vector memory
        context = retrieve_memory(user_input)

        response = chain.invoke({
            "input": user_input,
            "context": context
        })

        print("\n🏷️ Response:", response.content)

        # Save conversation to vector database
        save_to_memory(user_input)
        save_to_memory(response.content)

        print("\n🌍 Tokens:", cb.total_tokens)


# Run vector memory chain example
if __name__ == "__main__":
    vector_memory_chain()

