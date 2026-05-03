# LangChain basics examples - comprehensive demonstration of core concepts
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback

from langchain_core.messages import AIMessage

# Debug configuration - enable for detailed execution tracing
# set_debug(True)

# Load environment variables including API keys
load_dotenv()


# Model factory - centralized LLM configuration
def create_model():
    return ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.3,
        max_tokens=200,
        max_retries=2,
    )


# Helper: Extract token usage from individual AI responses
def print_token_usage(response):
    if isinstance(response, AIMessage):
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        print("\n🔬 Per-Call Token Usage:")
        print(f"Prompt Tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion Tokens: {usage.get('completion_tokens', 0)}")
        print(f"Total Tokens: {usage.get('total_tokens', 0)}")
    else:
        print("\n⚠️ No token metadata (response transformed by chain)")


# Helper: Display aggregate usage metrics from callback
def print_callback_usage(cb):
    print("\n🌍 Aggregate Usage (Callback):")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


# 🧪 1. Basic LLM invocation - direct model call
def simple_llm_call():
    llm = create_model()

    with get_openai_callback() as cb:
        response = llm.invoke(
            "I want to open a restaurant for Indian food. Suggest one fancy name. Only one name."
        )

        print("\n🍽️ Restaurant Name:", response.content)

        print_token_usage(response)   # per-call metrics
        print_callback_usage(cb)      # aggregate metrics


# 🧪 2. Prompt Template - structured prompts with variables
def simple_prompt_template(cuisine):
    llm = create_model()

    prompt_template = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest one fancy name. Only one name.",
    )

    formatted_prompt = prompt_template.format(cuisine=cuisine)

    with get_openai_callback() as cb:
        response = llm.invoke(formatted_prompt)

        print("\n🍽️ Restaurant Name:", response.content)

        print_token_usage(response)
        print_callback_usage(cb)

# Reusable prompt template for restaurant name generation
def gen_name_prompt_template(): 
    return PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a Restaurant for {cuisine} food. Suggest a fancy name. Give only one name."
        )

# Reusable prompt template for menu item generation
def gen_items_prompt_template(): 
    return PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return as a comma separator"
        )

# 🧪 3. LCEL Chain - modern pipe operator approach
def simple_chain(cuisine):
    llm = create_model()

    prompt = gen_name_prompt_template()

    chain = prompt | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"cuisine": cuisine})

        print("\n🍽️ Restaurant Name:", response.content)

        print_token_usage(response)   # per-call insight
        print_callback_usage(cb)      # full execution insight

# Simple sequence chain - single input/output pipeline
def simple_seq_chain(cuisine):
    llm = create_model()

    name_prompt = gen_name_prompt_template()
    items_prompt = gen_items_prompt_template()

    # Create sequential pipeline: name → format → menu items
    chain = name_prompt | llm | (lambda x: {"restaurant_name": x.content}) | items_prompt | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"cuisine": cuisine})

        print("\n🍽️ Final Output (Menu Items):", response.content)

        print_token_usage(response)
        print_callback_usage(cb)

# Advanced sequence chain - multi-step with intermediate output visibility
def base_seq_chain(cuisine):
    llm = create_model()

    name_prompt = gen_name_prompt_template()
    items_prompt = gen_items_prompt_template()

    # Step 1: Generate restaurant name
    def generate_name(inputs):
        response = (name_prompt | llm).invoke(inputs)
        return {
            "restaurant_name": response.content
        }

    # Step 2: Generate menu items using restaurant name
    def generate_items(inputs):
        response = (items_prompt | llm).invoke(inputs)
        return {
            "menu_items": response.content
        }

    with get_openai_callback() as cb:
        # Execute step 1
        step1 = generate_name({"cuisine": cuisine})
        print("\n🏷️ Generated Name:", step1["restaurant_name"])

        # Execute step 2 using step 1 output
        step2 = generate_items(step1)
        print("\n🍽️ Menu Items:", step2["menu_items"])

        print_callback_usage(cb)


# 🎯 Entry point - run basic LLM example
if __name__ == "__main__":
    simple_llm_call()
    # Uncomment to test other examples:
    # simple_prompt_template(cuisine='Indian')
    # simple_chain(cuisine='Indian')
    # simple_seq_chain(cuisine='Indian')
    # base_seq_chain(cuisine='Indian')
