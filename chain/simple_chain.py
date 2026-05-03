# LangChain imports for chains and callbacks
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model
from utils import langchain_helper as lc_helper

# 🔧 Debug (optional) Enable whenever you want to see in-detail execution
# set_debug(True)

# 🧪 LCEL Chain Example - Chain prompt template with LLM using pipe operator
def simple_chain(cuisine):
    # Create LLM instance
    llm = create_model()

    # Get prompt template from helper
    prompt = lc_helper.gen_name_prompt_template()

    # Create chain using LCEL (LangChain Expression Language)
    chain = prompt | llm

    # Execute chain with token tracking
    with get_openai_callback() as cb:
        response = chain.invoke({"cuisine": cuisine})

        print("\n🍽️ Restaurant Name:", response.content)

        # Display usage metrics
        lc_helper.print_token_usage(response)   # 🔬 exact call insight
        lc_helper.print_callback_usage(cb)      # 🌍 full execution insight



# 🎯 Run chain example with Indian cuisine
if __name__ == "__main__":
    simple_chain(cuisine='Indian')