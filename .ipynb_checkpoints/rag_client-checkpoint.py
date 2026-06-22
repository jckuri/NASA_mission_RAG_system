import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

import os
from chromadb.utils import embedding_functions


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        path for path in current_dir.iterdir()
        if path.is_dir() and ("chroma" in path.name.lower() or "db" in path.name.lower() or "vector" in path.name.lower())
    ]
    
    for chroma_dir in chroma_dirs:
        print(f'chroma_dir: {chroma_dir}')

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
        try:        
            # TODO: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path = str(chroma_dir), settings = Settings(anonymized_telemetry=False))
            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()
            # TODO: Loop through each collection found
            for collection in collections:
                collection_name = collection.name
                # TODO: Create unique identifier key combining directory and collection names
                backend_key = f"{chroma_dir.name}:{collection_name}"
                # Get collection count
                try:
                    count = collection.count()
                except Exception:
                    count = "unknown"
                # TODO: Build information dictionary containing:
                information = {
                    # TODO: Store directory path as string
                    "path": str(chroma_dir),
                    # TODO: Store collection name
                    "collection_name": collection_name,
                    # TODO: Create user-friendly display name
                    "display_name": f"{chroma_dir.name} / {collection_name} ({count} documents)",
                    # TODO: Get document count with fallback for unsupported operations
                    "count": str(count)
                }
                # TODO: Add collection information to backends dictionary
                backends[backend_key] = information
        
        # TODO: Handle connection or access errors gracefully
        except Exception as e:
            # TODO: Include error information in display name with truncation
            error_text = str(e)
            if len(error_text) > 80:
                error_text = error_text[:80] + "..."
            backend_key = f"{chroma_dir.name}:error"
            # TODO: Set appropriate fallback values for missing information
            backends[backend_key] = {
                "path": str(chroma_dir),
                "collection_name": "",
                # TODO: Create fallback entry for inaccessible directories
                "display_name": f"{chroma_dir.name} - inaccessible: {error_text}",
                "count": "unknown",
                "error": error_text,
            }
            
    # TODO: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system2(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    # TODO: Create a ChromaDB persistent client
    client = chromadb.PersistentClient(
        path = chroma_dir,
        settings = Settings(anonymized_telemetry=False)
    )
    # TODO: Return the collection with the collection_name
    collection = client.get_collection(name = collection_name)  
    success = True
    error = None
    return collection, success, error

def initialize_rag_system(chroma_dir: str, collection_name: str):
    try:
        openai_api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("CHROMA_OPENAI_API_KEY")
        )

        if not openai_api_key:
            return None, False, "Missing OPENAI_API_KEY"

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small",
        )

        client = chromadb.PersistentClient(
            path=chroma_dir
        )

        collection = client.get_collection(
            name=collection_name,
            embedding_function=openai_ef,
        )

        return collection, True, None

    except Exception as e:
        return None, False, str(e)


def retrieve_documents2(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None
    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}
    # TODO: Execute database query with the following parameters:
    results = collection.query(
        # TODO: Pass search query in the required format
        query_texts = [query],
        # TODO: Set maximum number of results to return
        n_results = n_results,
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where = where_filter
    )
    # TODO: Return query results to caller
    return results


def retrieve_documents3(collection, query: str, n_results: int = 3, mission_filter=None):
    query_embedding = get_embedding(query)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if mission_filter:
        query_args["where"] = {"mission": mission_filter}

    return collection.query(**query_args)


# def get_embedding(self, input_text: str) -> List[float]:
#     response = self.openai_client.embeddings.create(
#         model=self.embedding_model,
#         input=input_text
#     )

#     return response.data[0].embedding

def retrieve_documents(collection, query: str, n_results: int = 3, mission_filter=None):
    try:
        query_args = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if mission_filter:
            query_args["where"] = {"mission": mission_filter}

        return collection.query(**query_args)

    except Exception as e:
        raise RuntimeError(f"Error retrieving documents: {e}")


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    # TODO: Initialize list with header text for context section
    context_parts = ["Relevant context from the knowledge base:"]
    # TODO: Loop through paired documents and their metadata using enumeration
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}
        # TODO: Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown mission")
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = str(mission).replace("_", " ").title()
        # TODO: Extract source information from metadata with fallback value  
        source = metadata.get("source", "unknown source")
        # TODO: Extract category information from metadata with fallback value
        category = metadata.get("category", "general")
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category = str(category).replace("_", " ").title()
        # TODO: Create formatted source header with index number and extracted information
        source_header = f"\n[Source {index}] Mission: {mission} | Category: {category} | Source: {source}"
        # TODO: Add source header to context parts list
        context_parts.append(source_header)
        # TODO: Check document length and truncate if necessary
        max_chars = 1500
        if len(document) > max_chars:
            document = document[:max_chars].rstrip() + "..."
        # TODO: Add truncated or full document content to context parts list
        context_parts.append(document)
    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
