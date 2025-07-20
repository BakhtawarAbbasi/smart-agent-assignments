from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Setup Gemini API
gemini_api_key = "AIzaSyDikAKCtOAWIqKRNXXXUgPgrmJKSxrdUjg"

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # or gemini-2.0 if supported
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Tool 1: Capital Agent
CapitalAgent = Agent(
    name="CapitalAgent",
    instructions="You are a helpful assistant that tells the capital of a country. Only answer with the capital."
)

# Tool 2: Language Agent
LanguageAgent = Agent(
    name="LanguageAgent",
    instructions="You are a helpful assistant that tells the primary language of a country. Only answer with the language."
)

# Tool 3: Population Agent
PopulationAgent = Agent(
    name="PopulationAgent",
    instructions="You are a helpful assistant that tells the population of a country. Only answer with the approximate population."
)

# ✅ Orchestrator Agent: Uses all 3 tools
OrchestratorAgent = Agent(
    name="CountryInfoOrchestrator",
    instructions="""
You are an orchestrator agent that takes the name of a country from the user and uses three tool agents to gather:
1. The capital city
2. The primary language
3. The approximate population

Combine the answers into a single clear response like:
"The capital of {country} is {capital}, the main language spoken is {language}, and the population is around {population}."
""",
    handoffs=[CapitalAgent, LanguageAgent, PopulationAgent]
)

# Test prompt
prompt = "Tell me about Pakistan"
result = Runner.run_sync(OrchestratorAgent, prompt, run_config=config)

print("✅ Final Answer:\n", result.final_output)
print("\n📌 Last agent used:", result.last_agent)
