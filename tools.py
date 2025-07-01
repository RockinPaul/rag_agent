from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools import FunctionTool
import random
from huggingface_hub import list_models
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Wikipedia search tool
tool_spec = WikipediaToolSpec()


def truncated_wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a query and returns a truncated version of the first result."""
    # tool_spec.search_data returns a list of Document objects
    documents = tool_spec.search_data(query)
    if documents:
        content = documents[0].get_content()
        return content[:4000]  # Truncate to 4000 chars
    return "No result found."


search_tool = FunctionTool.from_defaults(truncated_wikipedia_search)

# Example usage:
# response = search_tool("What is a Large Language Model?")
# print(response)


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# Initialize the tool
weather_info_tool = FunctionTool.from_defaults(get_weather_info)


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(
            list_models(author=author, sort="downloads", direction=-1, limit=1)
        )

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)

# Example usage
print(hub_stats_tool("facebook"))  # Example: Get the most downloaded model by Facebook


async def main():
    # Initialize the OpenAI model
    openai_api_key = os.getenv("OPEN_AI_API_KEY")

    llm = OpenAI(
        model_name="gpt-4.1-nano-2025-04-14",
        temperature=0.7,
        max_tokens=100,
        api_key=openai_api_key,
    )
    # Create Alfred with all the tools
    alfred = AgentWorkflow.from_tools_or_functions(
        [search_tool, weather_info_tool, hub_stats_tool], llm=llm
    )

    # Example query Alfred might receive during the gala
    response = await alfred.run("What is Facebook and what's their most popular model?")

    print("🎩 Alfred's Response:")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
