from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools import FunctionTool
import random
from huggingface_hub import list_models

# Initialize the Wikipedia search tool
tool_spec = WikipediaToolSpec()

search_tool = FunctionTool.from_defaults(tool_spec.search_data)

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
