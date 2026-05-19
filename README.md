Multimodal RAG-Based Document Comparison System

This project implements a Multimodal Retrieval-Augmented Generation (RAG) pipeline to compare two versions of PDF documents and identify structural and semantic differences.
The system:
Parses PDFs using layout-aware document partitioning
Extracts text, tables, and images
Performs title-aware semantic chunking
Generates embeddings
Stores content in a vector database
Matches semantically similar sections
Detects added, removed, and modified content
Generates an AI-powered explanation of changes
The system is deployed as a Streamlit web application.


Architecture Overview

PDF v1 / PDF v2
        ↓
Unstructured Partitioning (hi_res)
        ↓
Element Classification (Text / Tables / Images)
        ↓
Title-aware Chunking
        ↓
Embedding Generation (Sentence Transformers)
        ↓
Vector Storage (ChromaDB)
        ↓
Semantic Chunk Matching
        ↓
Structural Difference Detection
        ↓
LLM-based Explanation (Ollama)


Technologies Used
Python 3.13.5
Streamlit
Unstructured
Sentence Transformers (MiniLM-L6-v2)
ChromaDB
LangChain
Ollama (Local LLM)


How to Run the Application
1️. Install Dependencies
pip install -r requirements.txt

2️. Start Ollama (Required for AI Explanation)
Make sure Ollama is running locally:
ollama pull llama3

3️. Launch the Web App
streamlit run app.py

