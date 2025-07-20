from dotenv import load_dotenv
import os
from agents import Agent, Runner


# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
assert api_key, "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env"

# identifier for Gemini
model_name = "litellm/gemini/gemini-2.5-flash"

product_recommender_agent = Agent(
    name="medicine-recommender",
    model=model_name,
    instructions="""
You are a product recommendation assistant for a smart pharmacy.
Your job is to suggest a suitable medicine or product based on the user's symptoms or needs.
Always explain why the product is recommended in simple terms.
Only suggest safe and over-the-counter medicines unless instructed otherwise.
"""
)
def main():
    prompt = input("💬 Enter symptoms or need: ")
    result = Runner.run_sync(product_recommender_agent, prompt)
    print("\n🤖 Recommendation:\n", result.final_output)

if __name__ == "__main__":
    main()
