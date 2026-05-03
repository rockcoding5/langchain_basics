# Memory sequence chain - multi-step conversation with context retention
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from create_model.create_model import create_model

# Debug configuration - enable for detailed execution tracing
# set_debug(True)


def memory_seq_chain(cuisine: str):
    # Create LLM instance
    llm = create_model()

    # Initialize memory store for conversation history
    chat_history = []

    # Create prompt template with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for restaurant ideas."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # Build chain with prompt and LLM
    chain = prompt | llm

    # Execute multi-step conversation with token tracking
    with get_openai_callback() as cb:

        # Step 1: Generate restaurant name
        user_input_1 = f"Suggest a fancy restaurant name for {cuisine} food."

        response_1 = chain.invoke({
            "input": user_input_1,
            "chat_history": chat_history
        })

        print("\n🏷️ Generated Name:", response_1.content)

        # Save conversation to memory
        chat_history.append(HumanMessage(content=user_input_1))
        chat_history.append(AIMessage(content=response_1.content))


        # Step 2: Generate menu items using previous context
        user_input_2 = "Suggest menu items for the above restaurant."

        response_2 = chain.invoke({
            "input": user_input_2,
            "chat_history": chat_history
        })

        print("\n🍽️ Menu Items:", response_2.content)


        # Step 3: Generate tagline demonstrating memory continuity
        user_input_3 = "Give a short tagline for this restaurant."

        response_3 = chain.invoke({
            "input": user_input_3,
            "chat_history": chat_history
        })

        print("\n✨ Tagline:", response_3.content)


        # Display aggregate token usage
        print("\n🌍 Aggregate Usage (Memory Chain):")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")


# 🎯 Run memory sequence chain example with Indian cuisine
if __name__ == "__main__":
    memory_seq_chain(cuisine="Indian")
