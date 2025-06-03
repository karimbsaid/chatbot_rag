# Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot that integrates LangChain, ChromaDB, and FastAPI to provide intelligent responses based on your custom data sources.

## 🧠 Overview

This project enables you to deploy a local AI chatbot capable of:

- Converting the extracted data into vector embeddings using `sentence-transformers`  
- Storing embeddings in ChromaDB for efficient retrieval  
- Utilizing LangChain to process queries and generate context-aware responses  
- Serving the chatbot via a FastAPI backend

## 🚀 Features

- **PDF Extraction**: Parses and extracts text from PDF documents  
- **Vector Embedding**: Transforms text into vector representations  
- **ChromaDB Integration**: Stores and retrieves embeddings efficiently  
- **LangChain Integration**: Generates answers using retrieved context  
- **FastAPI Backend**: Exposes all functionalities through API endpoints

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher  
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) (optional, for containerized deployment)

### Clone the Repository

```bash
git clone https://github.com/karimbsaid/chatbot_rag.git
cd chatbot_rag
```


### Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.

## 🐳 Docker Deployment (Optional)

To deploy the application using Docker:

```bash
docker-compose up --build
```

This will build the Docker image and start the containerized application.

## 📁 Project Structure

```
chatbot_rag/
├── chroma_db/           # ChromaDB-related files
├── models/              # Pre-trained models and embeddings
├── routes/              # API route definitions
├── services/            # Core logic for scraping, embedding, etc.
├── utils/               # Utility functions
├── .env                 # Environment variables
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── main.py              # FastAPI application entry point
├── pdf_extractor.py     # PDF extraction logic
├── requirements.txt     # Python dependencies
└── app_config.py        # Application configuration
```

## 📋 API Endpoints

The application exposes the following endpoints:

- **POST /store/from-file**: Extract and store structured sections from an uploaded PDF file
- **POST /ask/stream**: Submit a query to the chatbot and receive a streamed response

### POST /store/from-file

This endpoint processes a PDF file, extracts structured content based on font size and spacing, splits it into chunks, and stores the chunks as documents for retrieval.

**Form Data Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | File | Yes | — | The PDF file to be processed |
| heading_font_threshold | float | No | 12.0 | Minimum font size to detect section headings |
| debut_document | int | No | 0 | Page number to start processing from |
| space_threshold | float | No | 1.5 | Space threshold used to separate paragraphs/sections |

**Example Request using curl:**

```bash
curl -X POST http://127.0.0.1:8000/store/from-file \
  -F "file=@yourfile.pdf" \
  -F "heading_font_threshold=12.0" \
  -F "debut_document=0" \
  -F "space_threshold=1.5"
```

**Success Response:**

```json
{
  "message": "X sections stored"
}
```

Where X is the number of content sections successfully extracted and stored.

### POST /ask/stream

This endpoint submits a query to the chatbot and returns a streamed response based on the stored documents.

**Request Body:**

```json
{
  "question": "Your question here"
}
```

**Example Request using curl:**

```bash
curl -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

**Response:**

The response is streamed as plain text, allowing for real-time display of the AI-generated answer.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.


