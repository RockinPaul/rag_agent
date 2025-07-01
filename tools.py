from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools import FunctionTool

# Initialize the Wikipedia search tool
tool_spec = WikipediaToolSpec()

search_tool = FunctionTool.from_defaults(tool_spec.search_data)

# Example usage:
response = search_tool("What is a Large Language Model?")
print(response)

