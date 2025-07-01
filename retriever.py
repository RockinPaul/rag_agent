import datasets
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join(
            [
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}",
            ]
        ),
        metadata={"name": guest_dataset["name"][i]},
    )
    for i in range(len(guest_dataset))
]

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)


def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results[:3]])
    else:
        return "No matching guest information found."


# Initialize the tool
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)



# Initialize the OpenAI model
openai_api_key = os.getenv("OPEN_AI_API_KEY")

llm = OpenAI(
    model_name="gpt-4.1-nano-2025-04-14",
    temperature=0.7,
    max_tokens=100,
    api_key=openai_api_key,
)

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

async def main():
    # Example query Alfred might receive during the gala
    response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

    print("🎩 Alfred's Response:")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
