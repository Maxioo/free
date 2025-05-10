import os
from llm import LLM
from memory import MemoryManager
from typing import Optional, List, Dict, Any

def print_memories(memories: List[Dict[str, Any]]) -> None:
    """Print memories in a formatted way."""
    if not memories:
        print("No memories found.")
        return
        
    print("\nRelevant Memories:")
    for i, memory in enumerate(memories, 1):
        print(f"\n{i}. Content: {memory['content']}")
        if memory.get('metadata'):
            print(f"   Metadata: {memory['metadata']}")
        if memory.get('created_at'):
            print(f"   Created: {memory['created_at']}")
        if memory.get('relevance_score') is not None:
            print(f"   Relevance: {memory['relevance_score']:.2f}")


def print_profile(profile: Dict[str, Any]) -> None:
    """Print user profile in a formatted way."""
    print("\nUser Profile:")
    if not profile.get('recent_memories'):
        print("No recent memories found.")
        return
        
    print("\nRecent Memories:")
    for i, memory in enumerate(profile['recent_memories'], 1):
        print(f"\n{i}. Content: {memory['content']}")
        if memory.get('metadata'):
            print(f"   Metadata: {memory['metadata']}")
        if memory.get('created_at'):
            print(f"   Created: {memory['created_at']}")


def main():
    """Main function demonstrating LLM and Memory usage."""
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Create LLM instance with memory support
    llm = LLM.create_silicon_flow(
        memory_manager=memory_manager,
        user_id="demo_user"  # You can change this to any user ID
    )
    
    print("Welcome to the Chat Demo!")
    print("Available commands:")
    print("  /search <query> - Search memories")
    print("  /profile - View user profile")
    print("  /clear - Clear chat history")
    print("  /exit - Exit the chat")
    print("\nStart chatting (or use a command):")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input[1:].split(' ', 1)
                cmd = command[0].lower()
                
                if cmd == 'exit':
                    print("Goodbye!")
                    break
                    
                elif cmd == 'search' and len(command) > 1:
                    query = command[1]
                    print(f"\nSearching memories for: {query}")
                    memories = llm.search_memories(query)
                    print_memories(memories)
                    
                elif cmd == 'profile':
                    profile = llm.get_user_profile()
                    print_profile(profile)
                    
                elif cmd == 'clear':
                    if memory_manager.delete_user(llm.user_id):
                        print("Chat history cleared.")
                    else:
                        print("Failed to clear chat history.")
                        
                else:
                    print("Unknown command. Available commands:")
                    print("  /search <query> - Search memories")
                    print("  /profile - View user profile")
                    print("  /clear - Clear chat history")
                    print("  /exit - Exit the chat")
                continue
            
            # Regular chat with memory context
            print("\nAssistant: ", end="")
            for response in llm.chat_with_memories(user_input, stream=True):
                print(response, end="")
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
