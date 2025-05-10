# AI Chat with Memory

A Python-based chat application that combines LLM capabilities with persistent memory, allowing for context-aware conversations and memory management.

## Features

- 🤖 **Multiple LLM Providers**: Support for different LLM providers (OpenRouter, Silicon Flow)
- 🧠 **Memory Management**: Persistent memory storage using Mem0
- 🔍 **Memory Search**: Semantic search through conversation history
- 👤 **User Profiles**: Track and manage user-specific memories
- 💬 **Interactive Chat**: Command-line interface with memory-aware responses
- ⚡ **Streaming Responses**: Real-time response streaming for better user experience

## Prerequisites

- Python 3.8+
- OpenAI API key (for Mem0)
- Provider-specific API keys (OpenRouter, Silicon Flow)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `config.yaml` file in the project root:
```yaml
providers:
  open_router:
    api_key: "your-openrouter-api-key"
    api_base: "https://openrouter.ai/api/v1"
    model: "openai/gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 2000
    top_p: 1.0
    system_prompt: "You are a helpful AI assistant. Use the provided memories to give context-aware responses."

  silicon_flow:
    api_key: "your-siliconflow-api-key"
    api_base: "https://api.siliconflow.com/v1"
    model: "gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 2000
    top_p: 1.0
    system_prompt: "You are a helpful AI assistant. Use the provided memories to give context-aware responses."

memory:
  mem0:
    llm:
      provider: "litellm"
      config:
        api_key: "your-openai-api-key"
        openai_base_url: "https://api.openai.com/v1"
        model: "openai/THUDM/GLM-Z1-9B-0414"
        temperature: 0.7
        max_tokens: 4096
    embedder:
      provider: "openai"
      config:
        api_key: "your-openai-api-key"
        openai_base_url: "https://api.openai.com/v1"
        model: "BAAI/bge-m3"
        embedding_dims: 1024
    vector_store:
      provider: "qdrant"
      config:
        embedding_model_dims: 1024
    search_limit: 5
    context_limit: 3
```

## Usage

Run the chat application:
```bash
python main.py
```

### Available Commands

- `/search <query>` - Search through conversation memories
- `/profile` - View your conversation profile and recent memories
- `/clear` - Clear your chat history
- `/exit` - Exit the chat application

### Example Interaction

```
Welcome to the Chat Demo!
Available commands:
  /search <query> - Search memories
  /profile - View user profile
  /clear - Clear chat history
  /exit - Exit the chat

Start chatting (or use a command):

You: Hello! I'm interested in learning about AI agents.

Assistant: Hello! I'd be happy to help you learn about AI agents. AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve specific goals...

You: /search AI agents

Searching memories for: AI agents

Relevant Memories:
1. Content: Hello! I'm interested in learning about AI agents.
   Created: 2024-03-21T10:30:15
   Relevance: 0.95
```

## Project Structure

```
.
├── config.yaml           # Configuration file
├── main.py              # Main application entry point
├── llm.py              # LLM provider implementation
├── memory.py           # Memory management system
└── requirements.txt    # Project dependencies
```

### Key Components

- `LLM`: Handles chat completions and provider management
- `MemoryManager`: Manages memory operations and user profiles
- `MemoryConfig`: Configuration for the memory system
- `LLMConfig`: Configuration for LLM providers

## Configuration

### LLM Providers

The application supports multiple LLM providers through the `config.yaml` file. Each provider can be configured with:

- API key and base URL
- Model selection
- Temperature and token settings
- System prompt customization

### Memory System

The memory system (Mem0) is configured with:

- LLM settings for memory processing
- Embedding model configuration
- Vector store settings
- Search and context limits

## Development

### Adding a New Provider

1. Add provider configuration to `config.yaml`
2. Update the `LLM` class in `llm.py` to support the new provider
3. Add a factory method for the new provider

### Extending Memory Features

1. Modify `MemoryManager` in `memory.py`
2. Update the memory configuration in `config.yaml`
3. Add new memory operations as needed

## License

MIT License

Copyright (c) 2024 AI Chat with Memory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
