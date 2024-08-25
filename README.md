# LegalX Backend

## Overview

This is the backend for the LegalX project, an AI-powered legal assistant using RAG (Retrieval-Augmented Generation) technology. It's fine-tuned on specialized legal data and uses advanced language models to provide accurate and context-aware responses to legal queries.

## Tech Stack

- **Model**: Ollama Mistral
- **Framework**: FastAPI
- **Embedding**: Hugging Face embedding model
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Language Model Framework**: LangChain
- **Authentication**: Custom Auth system
- **Infrastructure**: AWS G5 instance

## Features

- RAG-based query answering system
- PDF document processing and indexing
- Secure API with custom authentication
- Scalable vector search using FAISS
- Integration with Ollama Mistral model for advanced language understanding

## Setup

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up environment variables (see `.env.example`)
4. Run the FastAPI server:
```
uvicorn main:app --reload
```

## Data Management

- Currently, PDF documents are stored in the `/data` folder
- Future implementation will use Amazon S3 for scalable storage

## API Endpoints

- `/ask`: POST request to submit a question and receive an AI-generated answer
- `/upload`: POST request to upload and process new PDF documents
- (Add other endpoints as necessary)

## Authentication

The backend uses a custom authentication system to secure the API. Ensure proper authentication headers are included in all requests.

## Development Status

- Chat feature is currently under development
- Authentication system is implemented to manage API access

## Future Enhancements

- Migration of document storage to Amazon S3
- Implementation of server-side actions for improved performance
- Continuous model fine-tuning with new legal data

## License

This project is licensed under the [MIT License](LICENSE).
