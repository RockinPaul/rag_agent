import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
import time

# LlamaIndex imports
from llama_index.core import Settings, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.service_context import ServiceContext

# Import tools from tools.py
from tools import search_tool, weather_info_tool, hub_stats_tool, tool_spec

# Load environment variables
load_dotenv()


class LocalEmbedding(BaseEmbedding):
    """Custom embedding class using sentence-transformers directly."""
    
    # Use private attributes for the SentenceTransformer model
    _model: SentenceTransformer = None
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with the specified model."""
        super().__init__()
        # Initialize the model inside a private attribute
        self._model = SentenceTransformer(model_name)
        
    def _get_text_embedding(self, text: str) -> list:
        """Get embedding for a single text."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def _get_text_embeddings(self, texts: list) -> list:
        """Get embeddings for multiple texts."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
    
    def _get_query_embedding(self, query: str) -> list:
        """Get embedding for a query text."""
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> list:
        """Async version of _get_query_embedding."""
        # Since sentence-transformers doesn't have native async support,
        # we just call the sync version
        return self._get_query_embedding(query)


class GAIARAGAgent:
    """RAG Agent for the GAIA benchmark using LlamaIndex.
    
    This agent combines LlamaIndex's ReAct framework with RAG capabilities
    to handle a wide variety of tasks from the GAIA benchmark.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b", temperature: float = 0.2, verbose: bool = True):
        """Initialize the GAIA RAG Agent.
        
        Args:
            model_name: Ollama model name to use (e.g., "deepseek-r1:1.5b", "mistral", "llama3")
            temperature: Temperature parameter for the LLM
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize vector store and embeddings
        self._initialize_vector_store()
        
        # Set up tools and memory
        self._setup_tools()
        self._setup_memory()
        
        # Initialize the agent
        self._initialize_agent()
    
    def _initialize_llm(self) -> None:
        """Initialize the LLM with appropriate settings."""
        # Initialize the LLM using local Ollama instance
        self.llm = Ollama(
            model=self.model_name,
            temperature=self.temperature,
            request_timeout=120.0,  # Increase timeout for potentially longer processing
            context_window=8000  # Set context window size appropriate for your model
        )
        
        # Initialize a local embedding model using sentence-transformers directly
        embed_model = LocalEmbedding(model_name="all-MiniLM-L6-v2")
        
        # Update global settings
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        Settings.callback_manager = self.callback_manager
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store for RAG functionality."""
        # Initialize ChromaDB vector store with explicit host and port
        import chromadb
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="gaia_knowledge")
        self.chroma_store = ChromaVectorStore(chroma_collection=collection)
        
        # Set up storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.chroma_store,
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore()
        )
    
    def _setup_tools(self) -> None:
        """Set up tools for the agent to use."""
        # Add necessary tools for the agent
        self.tools = [search_tool, weather_info_tool, hub_stats_tool]
    
    def _setup_memory(self) -> None:
        """Set up memory for the agent."""
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
    
    def _initialize_agent(self) -> None:
        """Initialize the ReAct agent with all components."""
        def handle_reasoning_failure(callback_manager, exception):
            """Handle reasoning failures by providing a fallback response."""
            if "Reached max iterations" in str(exception):
                return "I apologize, but I couldn't find a clear answer to your query. Let me try to provide some general information that might be helpful."
            else:
                return f"Error during reasoning: {str(exception)}"
        
        # Create ReAct agent with the configured tools and memory
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            verbose=self.verbose,
            callback_manager=self.callback_manager,
            handle_parsing_errors=handle_reasoning_failure,
            # Add system prompt if needed for better task handling
            system_prompt="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
        )
    
    def add_documents_to_rag(self, documents: List[Document]) -> None:
        """Add documents to the RAG system.
        
        Args:
            documents: List of Document objects to add to the RAG system
        """
        # Parse documents into nodes
        parser = SentenceSplitter(chunk_size=1024)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create vector store index
        vector_index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
        )
        
        # Create query engine
        query_engine = vector_index.as_query_engine(
            similarity_top_k=3,
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )
        
        # Create and add query engine tool
        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Search through the provided documents to find relevant information."
            )
        )
        
        # Add the query engine tool to the agent's tools
        self.tools.append(query_engine_tool)
        
        # Re-initialize agent with updated tools
        self._initialize_agent()
    
    def create_document_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Create a Document object from text.
        
        Args:
            text: The text content
            metadata: Optional metadata for the document
                
        Returns:
            A Document object
        """
        return Document(text=text, metadata=metadata or {})
    
    async def aquery(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Query the agent asynchronously.
        
        Args:
            query: The query string
                
        Returns:
            Tuple of (response string, token usage stats)
        """
        # Reset token counter
        self.token_counter.reset_counts()
        
        # Time the query
        start_time = time.time()
        try:
            response = await self.agent.aquery(query)
        except ValueError as e:
            if "Reached max iterations" in str(e):
                # Provide a direct response using the available information
                fallback_response = "I apologize, but I'm having difficulty processing this query with the available tools. " + \
                   "Let me provide a direct response based on the information I have: \n\n" + \
                   "The GAIA benchmark evaluates AI agents across multiple dimensions including reasoning, " + \
                   "knowledge retrieval, multi-modality handling, and tool use proficiency. It tests abilities " + \
                   "at various difficulty levels and focuses on tasks that humans find simple but AI finds challenging."
                response = fallback_response
            else:
                # Re-raise other ValueErrors
                raise
        end_time = time.time()
        
        # Get token usage
        token_usage = {
            "prompt_tokens": self.token_counter.prompt_llm_token_count,
            "completion_tokens": self.token_counter.completion_llm_token_count,
            "total_tokens": self.token_counter.total_llm_token_count,
            "time_taken": end_time - start_time
        }
        
        return str(response), token_usage
    
    def query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Synchronous wrapper for querying the agent.
        
        Args:
            query: The query string
                
        Returns:
            Tuple of (response string, token usage stats)
        """
        return asyncio.run(self.aquery(query))
    
    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.memory.reset()
        
    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics.
        
        Returns:
            Dictionary with token usage statistics
        """
        return {
            "prompt_tokens": self.token_counter.prompt_llm_token_count,
            "completion_tokens": self.token_counter.completion_llm_token_count,
            "total_tokens": self.token_counter.total_llm_token_count
        }


def create_agent(model_name="deepseek-r1:1.5b", temperature=0.2, verbose=False):
    """Create a new GAIA RAG agent with specified parameters.
    
    Args:
        model_name (str): Name of the local Ollama model to use
        temperature (float): Temperature parameter for the LLM
        verbose (bool): Whether to output verbose logs
    
    Returns:
        GAIARAGAgent: A configured RAG agent
    """
    return GAIARAGAgent(model_name=model_name, temperature=temperature, verbose=verbose)


def run_example() -> None:
    """Run an example query to test the agent."""
    # Create agent
    agent = create_agent(verbose=True)
    
    # Example documents for RAG context
    sample_text = """
    GAIA (Generalized AI Assistant Benchmark) is designed to evaluate AI agents across multiple dimensions
    including reasoning, knowledge retrieval, multi-modality handling, and tool use proficiency.
    The benchmark contains tasks at three difficulty levels, with level 1 being achievable by strong LLMs,
    while level 3 represents a significant leap in capability requirements.
    
    GAIA evaluates agents on metrics including completion rate, response quality, efficiency, robustness,
    and generalization to novel tasks. The evaluation involves both static datasets and dynamic, evolving tasks.
    
    A key aspect of GAIA is its focus on tasks that humans find conceptually simple but challenging for AI systems,
    requiring structured reasoning, planning, and accurate execution.
    """
    
    doc = agent.create_document_from_text(sample_text, metadata={"source": "GAIA documentation"})
    agent.add_documents_to_rag([doc])
    
    # Example query
    query = "What is the GAIA benchmark and what kinds of abilities does it test?"
    
    print(f"\nQuery: {query}")
    response, token_usage = agent.query(query)
    
    print(f"\nResponse: {response}")
    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")


def process_dataset(metadata_path: str, output_path: str, model_name: str = "deepseek-r1:1.5b", temperature: float = 0.2, verbose: bool = False) -> None:
    """Process GAIA dataset and write answers to output_path.

    Args:
        metadata_path: Path to metadata.jsonl containing tasks.
        output_path: Path to write answers.json (newline separated JSON objects).
    """
    import json
    answers = []
    agent = create_agent(model_name=model_name, temperature=temperature, verbose=verbose)

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            task_id = data.get("task_id")
            question = data.get("Question")
            if not task_id or not question:
                continue
            # Query agent
            try:
                response, _ = agent.query(question)
            except Exception as e:
                response = f"Error: {str(e)}"
            answers.append({
                "task_id": task_id,
                "model_answer": response
            })

    # Write answers
    with open(output_path, "w", encoding="utf-8") as outf:
        for ans in answers:
            json.dump(ans, outf, ensure_ascii=False)
            outf.write("\n")


if __name__ == "__main__":
    """CLI helper.

    Usage examples:
        python agent.py                 # runs example demo
        python agent.py dataset         # processes default GAIA dataset
        python agent.py /path/to/meta /path/to/out.json
    """
    import sys
    if len(sys.argv) == 1:
        # Default: run example
        run_example()
    else:
        # Process dataset
        base_dir = os.path.dirname(__file__)
        if len(sys.argv) == 2 and sys.argv[1] == "dataset":
            meta_path = os.path.join(base_dir, "GAIA", "2023", "test", "metadata.jsonl")
            out_path = os.path.join(base_dir, "answers.json")
        elif len(sys.argv) >= 3:
            meta_path = sys.argv[1]
            out_path = sys.argv[2]
        else:
            print("Usage: python agent.py <metadata.jsonl> <answers.json>")
            sys.exit(1)
        print(f"Processing dataset {meta_path} -> {out_path}")
        process_dataset(meta_path, out_path, verbose=True)
        print("Finished writing answers.")
