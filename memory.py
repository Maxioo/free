from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from pathlib import Path
import yaml
from mem0 import Memory as Mem0Memory


class MemoryConfig(BaseModel):
    """Configuration for Mem0 memory system."""
    llm: Dict[str, Any] = {
        "provider": "litellm",
        "config": {
            "api_key": None,
            "openai_base_url": None,
            "model": "openai/THUDM/GLM-Z1-9B-0414",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
    }
    embedder: Dict[str, Any] = {
        "provider": "openai",
        "config": {
            "api_key": None,
            "openai_base_url": None,
            "model": "BAAI/bge-m3",
            "embedding_dims": 1024,
        }
    }
    vector_store: Dict[str, Any] = {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1024,
        }
    }
    search_limit: int = 5
    context_limit: int = 3


class MemoryManager:
    """Memory management system using Mem0."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        """
        Initialize memory manager with configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in current directory
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get memory configuration
        memory_config = config.get('memory', {}).get('mem0', {})
        self.config = MemoryConfig(**memory_config)
        
        # Initialize Mem0 client with configuration
        self.memory = Mem0Memory.from_config({
            "llm": self.config.llm,
            "embedder": self.config.embedder,
            "vector_store": self.config.vector_store
        })
            
    def add_user(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add or get a user."""
        # Mem0 doesn't require explicit user creation
        return user_id
        
    def add_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a conversation to user's memory."""
        # Add conversation to memory with metadata
        memory_id = self.memory.add(
            messages,
            user_id=user_id,
            metadata=metadata or {}
        )
        return memory_id
        
    def get_user_profile(self, user_id: str, as_json: bool = True) -> Dict[str, Any]:
        """Get user's profile information."""
        # Get recent memories as profile
        memories = self.memory.search(
            "summary",
            user_id=user_id,
            limit=self.config.search_limit
        )
        return {
            "recent_memories": [
                {
                    "content": m["memory"],
                    "metadata": m.get("metadata", {}),
                    "created_at": m.get("timestamp")
                }
                for m in memories["results"]
            ]
        }
        
    def get_context(
        self,
        user_id: str,
        max_token_size: int = 500,
        prefer_topics: Optional[List[str]] = None
    ) -> str:
        """Get relevant context for a user."""
        # Get relevant memories
        memories = self.memory.search(
            "summary",
            user_id=user_id,
            limit=self.config.context_limit
        )
        
        # Format memories with metadata if available
        memory_str = "\n".join(
            f"- {m['memory']} (Metadata: {m.get('metadata', {})})"
            for m in memories["results"]
        )
        return f"Recent context:\n{memory_str}"
        
    def search_memories(
        self,
        user_id: str,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search user's memories."""
        # Use Mem0's semantic search
        results = self.memory.search(
            query,
            user_id=user_id,
            limit=min(max_results, self.config.search_limit)
        )
        return [
            {
                "content": m["memory"],
                "metadata": m.get("metadata", {}),
                "created_at": m.get("timestamp"),
                "relevance_score": m.get("score", 0.0)
            }
            for m in results["results"]
        ]
        
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their memories."""
        try:
            # Clear all memories for the user
            self.memory.clear(user_id=user_id)
            return True
        except:
            return False 