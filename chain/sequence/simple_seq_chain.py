# LangChain imports for sequential chains and callbacks
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model
from utils import langchain_helper as lc_helper

# 🔧 Debug (optional) Enable whenever you want to see in-detail execution
# set_debug(True)

# 🧪 Simple Sequence Chain - Multi-step pipeline with single input/output
def simple_seq_chain(cuisine):
    # Create LLM instance
    llm = create_model()

    # Get prompt templates for name generation and menu items
    name_prompt = lc_helper.gen_name_prompt_template()
    items_prompt = lc_helper.gen_items_prompt_template()

    # Create sequential pipeline: name → format → menu items
    chain = name_prompt | llm | (lambda x: {"restaurant_name": x.content}) | items_prompt | llm

    # Execute sequence with token tracking
    with get_openai_callback() as cb:
        response = chain.invoke({"cuisine": cuisine})

        print("\n🍽️ Final Output (Menu Items):", response.content)

        # Display usage metrics
        lc_helper.print_token_usage(response)
        lc_helper.print_callback_usage(cb)

# 🎯 Run sequence chain example with Indian cuisine
if __name__ == "__main__":
    simple_seq_chain(cuisine='Indian')
