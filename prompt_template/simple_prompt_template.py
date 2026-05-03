# LangChain imports for prompt templates and model interaction
from langchain_core.globals import set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from create_model.create_model import create_model
from utils import langchain_helper as lc_helper

# 🔧 Debug (optional) Enable whenever you want to see in-detail execution
# set_debug(True)


# 🧪 Prompt Template Example - Generate restaurant names using structured prompts
def simple_prompt_template(cuisine):
    # Create LLM instance
    llm = create_model()

    # Define prompt template with cuisine variable
    prompt_template = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest one fancy name. Only one name.",
    )

    # Format prompt with cuisine value
    formatted_prompt = prompt_template.format(cuisine=cuisine)

    # Generate response with token tracking
    with get_openai_callback() as cb:
        response = llm.invoke(formatted_prompt)

        print("\n🍽️ Restaurant Name:", response.content)

        # Display usage metrics
        lc_helper.print_token_usage(response)
        lc_helper.print_callback_usage(cb)


# 🎯 Run example with Indian cuisine
if __name__ == "__main__":
    simple_prompt_template(cuisine='Indian')
