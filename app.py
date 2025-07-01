# Import necessary libraries
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
import os
from dotenv import load_dotenv

load_dotenv()

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
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)


async def main():
    # query = "Tell me about Lady Ada Lovelace. What's her background?"
    # response = await alfred.run(query)

    # print("🎩 Alfred's Response:")
    # print(response.response.blocks[0].text)

    # query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
    # response = await alfred.run(query)

    # print("🎩 Alfred's Response:")
    # print(response)

    # query = "One of our guests is from Google. What can you tell me about their most popular model?"
    # response = await alfred.run(query)

    # print("🎩 Alfred's Response:")
    # print(response)

    # query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
    # response = await alfred.run(query)

    # print("🎩 Alfred's Response:")
    # print(response)

    # Remembering state
    ctx = Context(alfred)

    # First interaction
    response1 = await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
    print("🎩 Alfred's First Response:")
    print(response1)

    # Second interaction (referencing the first)
    response2 = await alfred.run("What projects is she currently working on?", ctx=ctx)
    print("🎩 Alfred's Second Response:")
    print(response2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
