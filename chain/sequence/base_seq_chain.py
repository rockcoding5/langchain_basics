# 🧠 Sequential Chain (Modern Implementation)
# Demonstrates multi-step reasoning without deprecated LLMChain APIs

from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model
from utils import langchain_helper as lc_helper
from utils.RestaurantOutput import RestaurantOutput


def base_seq_chain(cuisine: str) -> RestaurantOutput:
    """
    Modern sequential chain implementation:
    1. Generate restaurant name
    2. Generate menu items using previous output
    3. Return structured response

    Replaces old SequentialChain with explicit control
    """

    # Create LLM instance
    llm = create_model()

    # Get prompt templates for name and menu generation
    name_prompt = lc_helper.gen_name_prompt_template()
    items_prompt = lc_helper.gen_items_prompt_template()

    # Step 1: Generate restaurant name
    def generate_name(inputs):
        response = (name_prompt | llm).invoke(inputs)
        return {"restaurant_name": response.content}

    # Step 2: Generate menu using Step 1 output
    def generate_items(inputs):
        response = (items_prompt | llm).invoke(inputs)
        return {"menu_items": response.content}

    # Execute sequential chain with token tracking
    with get_openai_callback() as cb:

        # Generate restaurant name
        step1 = generate_name({"cuisine": cuisine})
        print("\n🏷️ Generated Name:", step1["restaurant_name"])

        # Generate menu items using restaurant name
        step2 = generate_items(step1)
        print("\n🍽️ Menu Items:", step2["menu_items"])

        # Create final structured output using Pydantic model
        final_output = RestaurantOutput(
            restaurant_name=step1["restaurant_name"],
            menu_items=step2["menu_items"]
        )

        # Display usage metrics
        lc_helper.print_callback_usage(cb)

        return final_output