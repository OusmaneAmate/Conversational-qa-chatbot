# Conversational Q&A Chatbot

Ask questions about your PDF documents. The PDFs are split into chunks, embedded with OpenAI and stored in Pinecone. Each question retrieves the most relevant chunks, and an OpenAI chat model answers from them.

## Project structure

```
Conversational_Q&A_Chatbot/
├── chatbot/            # Reusable code
│   ├── config.py       # Settings and API keys (read from .env)
│   ├── ingest.py       # Load PDFs → split → store in Pinecone
│   └── qa.py           # Retrieve chunks → answer questions
├── documents/          # Put your PDF files here
├── notebooks/
│   └── test.ipynb      # Step-by-step exploration
├── .env.example        # Template for your .env file
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies into the environment:
   ```powershell
   .\venv\python.exe -m pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY` and `PINECONE_API_KEY`.

## Usage

Run these from the project folder.

Store the PDFs in `documents/` in Pinecone. Run this again after adding or changing PDFs:
```powershell
.\venv\python.exe -m chatbot.ingest
```

Ask a single question:
```powershell
.\venv\python.exe -m chatbot.qa "What are the main priorities of this budget?"
```

Or start an interactive session:
```powershell
.\venv\python.exe -m chatbot.qa
```

## Configuration

Optional settings in `.env`:

| Variable | Default |
|---|---|
| `PINECONE_INDEX_NAME` | `langchainvector` |
| `LLM_MODEL` | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` |

If you change the embedding model to one with a different vector size, you also have to recreate the Pinecone index.
