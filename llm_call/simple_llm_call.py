# LangChain imports for LLM calls and callbacks
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback

from create_model.create_model import create_model
from utils import langchain_helper as lc_helper

# Debug configuration - enable for detailed execution tracing
# set_debug(True)

# 🧪 Basic LLM invocation - direct model call example
def simple_llm_call():
    # Create LLM instance
    llm = create_model()

    # Execute with token tracking
    with get_openai_callback() as cb:
        response = llm.invoke(
            "I want to open a restaurant for Indian food. Suggest one fancy name. Only one name."
        )

        print("\n🍽️ Restaurant Name:", response.content)

        # Display usage metrics
        lc_helper.print_token_usage(response)   # per-call metrics
        lc_helper.print_callback_usage(cb)      # aggregate metrics


# 🎯 Run simple LLM call example
if __name__ == "__main__":
    simple_llm_call()
