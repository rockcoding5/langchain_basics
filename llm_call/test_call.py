# Direct OpenAI API test - basic client usage
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables including API keys
load_dotenv()

# Optional: Debug API key (commented out)
# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

def test_call():
    # Create OpenAI client
    client = OpenAI()

    # Make chat completion request
    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )

    # Print response content
    print(resp.choices[0].message.content)

# 🎯 Run OpenAI API test
if __name__ == "__main__":
    test_call()
