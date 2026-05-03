# LangChain utility functions for token tracking and prompt templates
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from chain.sequence import base_seq_chain as sc

# Extract token usage from individual AI responses
def print_token_usage(response):
    if isinstance(response, AIMessage):
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        print("\n🔬 Per-Call Token Usage:")
        print(f"Prompt Tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion Tokens: {usage.get('completion_tokens', 0)}")
        print(f"Total Tokens: {usage.get('total_tokens', 0)}")
    else:
        print("\n⚠️ No token metadata (response transformed by chain)")


# Display per-step token usage for agent responses
def print_agent_token_usage(response):
    print("\n🔬 Per-Step Token Usage:")

    for i, msg in enumerate(response["messages"]):
        if msg.type == "ai":
            usage = msg.response_metadata.get("token_usage", {})

            print(f"\nStep {i}:")
            print(f"Prompt Tokens: {usage.get('prompt_tokens', 0)}")
            print(f"Completion Tokens: {usage.get('completion_tokens', 0)}")
            print(f"Total Tokens: {usage.get('total_tokens', 0)}")


# Display aggregate usage metrics from callback
def print_callback_usage(cb):
    print("\n🌍 Aggregate Usage (Callback):")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

# Generate reusable prompt template for restaurant names
def gen_name_prompt_template():
    return PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a Restaurant for {cuisine} food. Suggest a fancy name. Give only one name."
    )

# Generate reusable prompt template for menu items
def gen_items_prompt_template():
    return PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return as a comma separator"
    )


# Get restaurant name and menu items using sequence chain
def get_rest_name_item(cuisine):
    return sc.base_seq_chain(cuisine)

# Extract final answer from agent response
def get_final_answer(response) -> str:
    for msg in reversed(response["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "No valid response generated."

