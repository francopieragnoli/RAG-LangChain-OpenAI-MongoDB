# RAG with OpenAI, LangChain and MongoDB
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain embeddings and MongoDB as a vector database. The system processes PDF documents, splits the text into coherent chunks of up to 256 characters, stores them in MongoDB, and retrieves relevant chunks based on a prompt. The retrieved chunks are then sent to the OpenAI API to generate a final, context-aware response.

## Features
PDF Processing: Extracts text from PDF files and splits it into coherent chunks without interrupting sentences.

Vector Storage: Stores text chunks and their embeddings in MongoDB.

Similarity Search: Finds relevant chunks using cosine similarity.

OpenAI Integration: Generates final responses using the OpenAI API based on retrieved chunks.

Assertiveness Threshold: Filters retrieved chunks based on a similarity threshold.

## Requirements
Python 3.7+
MongoDB
LangChain
PyPDF2
OpenAI 
Python

## Notes
### Coherent Text Division:
The split_text_into_chunks function divides the text into sentences and ensures that each chunk ends in a period, respecting the 256 character limit.
### Chunk Filtering:
Only chunks with similarity above the defined assertiveness threshold are concatenated to form the context to be sent to OpenAI.
### OpenAI Answers:
If there are no chunks with similarity above the threshold, a default message is returned.
