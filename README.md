# GAIA Benchmark RAG Agent

This repository contains a Retrieval-Augmented Generation (RAG) agent built with LlamaIndex that is designed to pass the GAIA (Generalized AI Assistant) benchmark.

## Overview

The agent combines several key capabilities:
- ReAct framework for reasoning and action cycles
- RAG capabilities for knowledge retrieval and synthesis
- Multiple tool integration (Wikipedia, weather info, Hugging Face stats)
- Memory management for contextual conversations
- Performance tracking and metrics

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

The agent is implemented in `agent.py` and can be used as follows:

```python
from agent import create_agent

# Create the agent with default parameters (gpt-4.1-nano-2025-04-14)
agent = create_agent(verbose=True)

# Add documents to the RAG system if needed
doc = agent.create_document_from_text("Your text here", metadata={"source": "Source"})
agent.add_documents_to_rag([doc])

# Query the agent
response, token_usage = agent.query("Your question here")
print(f"Response: {response}")
```

## Running the Example

To run the built-in example:

```bash
python agent.py
```

This will execute a simple test query to demonstrate the agent's capabilities.

## Testing Against GAIA Benchmark

To test this agent against the GAIA benchmark:

1. First, ensure your environment is properly set up with all dependencies installed
2. The agent is designed to work with the GAIA benchmark's evaluation framework
3. You can integrate this agent with the official GAIA evaluation protocol

## Architecture

The agent uses LlamaIndex's ReActAgent as its foundation and enhances it with:

1. Vector storage for document retrieval
2. Tool integration for external information access
3. Memory management for conversation context
4. Response synthesis using tree summarization

## Features

- **Multi-tool Integration**: Uses Wikipedia, weather, and Hugging Face tools
- **RAG Capabilities**: Document indexing, retrieval, and synthesis
- **Conversation Memory**: Maintains context across exchanges
- **Performance Monitoring**: Tracks token usage and response times
- **Async Support**: Provides both synchronous and asynchronous query interfaces
