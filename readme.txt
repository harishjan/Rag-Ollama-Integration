# RAG-Ollama-Integration

## Project Overview

RAG-Ollama-Integration is a Python project that demonstrates how to combine Retrieval-Augmented Generation (RAG) techniques with the Ollama large language model API. The project retrieves external data, creates a context using a simple vector store, and queries the Ollama LLM to generate responses.

## Features

- Integration with the Ollama LLM API
- Retrieval of external data for context enrichment
- Simple in-memory vector store for context similarity search
- Modular handler class (`RAGOllamaHandler`) for workflow orchestration

## Project Structure

```
Rag-Ollama-Integration/
│
├── main.py                  # Entry point for running the project
├── ragollama.py             # Contains RAGOllamaHandler and vector store logic
├── ollama_integration.py    # (Optional) Additional Ollama integration utilities
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Rag-Ollama-Integration.git
   cd Rag-Ollama-Integration
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```sh
   python main.py
   ```

## Usage

- Edit `main.py` to customize prompts or workflow.
- Use the `RAGOllamaHandler` class in `ragollama.py` to interact with Ollama and the vector store.

## Configuration

- The Ollama API endpoint defaults to `http://localhost:11434/api/generate`. You can change this in the class constructor.

## License

This project is licensed under the MIT License.

##

Harish Janardhanan,harishjan@gmail.com
