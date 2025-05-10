from typing import List, Dict, Any, Optional, Generator
from pydantic import BaseModel
import yaml
import os
from pathlib import Path
from memory import MemoryManager
import uuid
from openai import OpenAI


class Message(BaseModel):
    """Message model for chat interactions."""
    role: str
    content: str


class LLMConfig(BaseModel):
    """Configuration for LLM."""
    api_key: str
    api_base: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    system_prompt: str = "You are a helpful AI assistant. Use the provided memories to give context-aware responses."


class LLM:
    """LLM class for handling chat completions with memory support."""
    
    def __init__(
        self,
        provider: str = "open_router",
        memory_manager: Optional[MemoryManager] = None,
        user_id: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize LLM with optional memory support.
        
        Args:
            provider: The provider to use (e.g., 'open_router', 'silicon_flow')
            memory_manager: Optional memory manager for context-aware responses
            user_id: Optional user identifier for memory operations
            config_path: Optional path to config file
        """
        self.provider = provider
        self.memory_manager = memory_manager
        self.user_id = user_id or str(uuid.uuid4())
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get provider configuration
        if provider not in config.get('providers', {}):
            raise ValueError(f"Provider '{provider}' not found in configuration")
            
        provider_config = config['providers'][provider]
        self.config = LLMConfig(**provider_config)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
            
    def chat_with_memories(
        self,
        message: str,
        user_id: Optional[str] = None,
        store_memory: bool = True,
        stream: bool = False
    ) -> Generator[str, None, None]:
        """
        Chat with memories, retrieving relevant context and storing the conversation.
        
        Args:
            message: User's message
            user_id: Optional user identifier (uses instance user_id if not provided)
            store_memory: Whether to store the conversation in memory
            stream: Whether to stream the response
            
        Yields:
            Response chunks if streaming, otherwise yields the complete response
            
        Raises:
            ValueError: If memory_manager is not initialized
        """
        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")
            
        user_id = user_id or self.user_id
        
        # Retrieve relevant memories
        relevant_memories = self.memory_manager.memory.search(
            query=message,
            user_id=user_id,
            limit=self.memory_manager.config.context_limit
        )
        
        # Format memories for context
        memories_str = "\n".join(
            f"- {entry['memory']} (Relevance: {entry.get('score', 0.0):.2f})"
            for entry in relevant_memories["results"]
        )
        
        # Prepare messages with context
        system_prompt = f"{self.config.system_prompt}\n\nRelevant Memories:\n{memories_str}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Generate response
        if stream:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            collected_response = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response.append(content)
                    yield content
                    
            # Store the complete conversation if requested
            if store_memory:
                messages.append({"role": "assistant", "content": "".join(collected_response)})
                self.memory_manager.memory.add(messages, user_id=user_id)
        else:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            assistant_response = response.choices[0].message.content
            
            # Store the conversation if requested
            if store_memory:
                messages.append({"role": "assistant", "content": assistant_response})
                self.memory_manager.memory.add(messages, user_id=user_id)
                
            yield assistant_response
            
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Generator[str, None, None]:
        """
        Generate a chat completion without memory context.
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            
        Yields:
            Response chunks if streaming, otherwise yields the complete response
        """
        if stream:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            yield response.choices[0].message.content
            
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user's profile information."""
        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")
        return self.memory_manager.get_user_profile(self.user_id)
        
    def search_memories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search user's memories."""
        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")
        return self.memory_manager.search_memories(self.user_id, query, max_results)

    @classmethod
    def create_silicon_flow(
        cls,
        memory_manager: Optional[MemoryManager] = None,
        user_id: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> 'LLM':
        """Create an LLM instance configured for Silicon Flow."""
        return cls(
            provider="silicon_flow",
            memory_manager=memory_manager,
            user_id=user_id,
            config_path=config_path
        )

    @classmethod
    def create_open_router(
        cls,
        memory_manager: Optional[MemoryManager] = None,
        user_id: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> 'LLM':
        """Create an LLM instance configured for OpenRouter."""
        return cls(
            provider="open_router",
            memory_manager=memory_manager,
            user_id=user_id,
            config_path=config_path
        ) 