from dotenv import load_dotenv
import os
from agents import Agent, Runner

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Gemini model via litellm
model_name = "litellm/gemini/gemini-2.5-flash"

# Agent 1: Mood Detector
mood_checker = Agent(
    name="mood-checker",
    model=model_name,
    instructions="""
You are a mood analyzer.
Your job is to read the user's message and respond with only one word: their mood.
Examples:
- If someone says "I'm feeling down today", respond with: sad
- If someone says "Life is great!", respond with: happy
- If they say "I'm under a lot of pressure", respond with: stressed
Just return one mood word only: happy, sad, stressed, angry, excited, etc.
"""
)

# Agent 2: Activity Suggestion
activity_suggester = Agent(
    name="activity-suggester",
    model=model_name,
    instructions="""
You are a wellness assistant.
If a user is feeling sad or stressed, suggest a healthy activity to improve their mood.
Keep it short and friendly. Mention why it's helpful.
Examples:
- Take a short walk outside – fresh air can clear your mind.
- Try deep breathing for 5 minutes – it helps reduce stress.
"""
)

async def main():
    user_input = input("💬 How are you feeling today?\n")

    # Step 1: Analyze mood
    mood_result = await Runner.run(mood_checker, user_input)
    mood = mood_result.final_output.strip().lower()

    print(f"\n🧠 Detected Mood: {mood}")

    # Step 2: If sad/stressed, suggest activity
    if mood in ["sad", "stressed"]:
        suggestion_result = await Runner.run(activity_suggester, f"I’m feeling {mood}.")
        print("\n💡 Suggested Activity:\n", suggestion_result.final_output)
    else:
        print("\n😊 Glad to hear that! Keep shining! ✨")

# Async runner
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
