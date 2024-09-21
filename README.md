# KareXpert-RAG-Chatbot

This chatbot is a context-aware chatbot built using a RAG (Retrieval-Augmented Generation) architecture. It integrates MongoDB Atlas vector search for document retrieval, the SentenceTransformer model for generating vector embeddings, and Google's Gemma-2b-it LLM for natural language responses.

It is designed to search through a MongoDB collection, retrieve relevant documents, and generate coherent responses to user queries using the information from the database. 

This project was developed during an internship at KareXpert and serves as an example of building an intelligent chatbot that can retrieve and process large amounts of data from MongoDB, combine it with LLM, and respond to user queries in an accurate and contextually appropriate manner.

Key Features : 

    - RAG Architecture: Combines document retrieval from MongoDB with large language model text generation for more insightful and data-driven responses.
    
    - MongoDB Atlas Vector Search: Leverages vector search index to retrieve relevant documents based on user queries.
    
    - Embedding Generation: Uses the SentenceTransformer (gte-small) model to generate vector embeddings for fields in the database.
    
    - Text Generation: Utilizes Google's Gemma-2b-it for generating detailed and human-like responses.

Project Structure


├── app/

│   ├── api.py               # FastAPI server for managing API endpoints

│   ├── bot.py               # Main file with the logic for embeddings, search, and LLM integration

├── requirements.txt         # Required Python libraries

├── sample_postman.txt       # Sample Postman body formats for testing APIs

├── README.md                # Project documentation


To integrate the LLM model we have 3 methods:

1. By Using Hugging Face Inference API to generate responses via the google/gemma-2-2b-it inference API

2. By Using Hugging Face pipeline method 

3. By directly loading the google/gemma-2b-it model from transformers library using AutoTokenizer, AutoModelForCausalLM method

I have used the 1st method which is the Hugging Face Inference API method using the Hugging Face Token

How to Set Up:

Prerequisites:

Before you start, ensure that you have:

    Python 3.8+ installed.

    Required Python libraries
    
    MongoDB Atlas account with a database and collections.
    
    Hugging Face API Token for accessing the Gemma model.
    
    Postman (optional but recommended for testing the API).

Clone the repository:

https://github.com/hriday-sehgal/KareXpert-RAG-Chatbot.git

cd bot

1. Install Dependencies

Use the following command to install the required Python libraries:

pip install -r requirements.txt

The libraries used in this project are listed in requirements.txt and include:

    fastapi
    
    pymongo
    
    sentence-transformers
    
    huggingface-hub

2. Setup Environment Variables

Make sure to replace your MongoDB URI and Hugging Face Token for security reasons. These sensitive data must be stored in a separate file (e.g., key_param.py) or as environment variables.

python - key_param.py file:

MONGO_URI = "your_mongodb_connection_string"

HF_TOKEN = "your_hugging_face_token"

NOTE: This code is designed to work with user-provided credentials, so make sure you use your own values when running the project.

3. Run the Application

You can run the FastAPI server using the following command:

uvicorn api:app --reload

This will start the FastAPI server, and you can access it at http://127.0.0.1:8000.

How to Use the bot:

1. Training the Bot

To train the bot on a MongoDB collection and generate vector embeddings, you can send a POST request to the /trainbot endpoint.

Sample Postman API Body (for movies collection) (MongoDB sample_mflix database is used):

Method: POST

Body:

json

{

    "collection_name": "movies",
    "fields": ["title", "plot", "fullplot"],
    "path": "embedding",
    "num_dimensions": 384,
    "similarity": "cosine"
}

2. Getting a Response

To query the bot and get a response generated using MongoDB context and the Gemma LLM, send a POST request to the /response endpoint.

Sample Postman API Body (for movies collection):

Method: POST

Body:

json

{

    "collection_name": "movies",
    "query": "What is the best romantic movie to watch and why?"
}

Sample Postman Requests

You can find sample body formats for Postman testing in the sample_postman.txt file. These requests demonstrate how to interact with the chatbot via API.

Folder Structure & Code Breakdown

    api.py: Contains all the FastAPI routes, including /trainbot for training the model and /response for generating chatbot responses.
    
    bot.py: Handles MongoDB connection, embedding generation, vector search, and response generation using the Gemma model.

    requirements.txt: Required Python libraries
    
    sample_postman.txt : Sample Postman body formats for testing APIs

Detailed Steps for Running the Project:

1. Connect to MongoDB: The bot connects to MongoDB Atlas to retrieve data from the specified collection. Ensure your collection contains documents with fields you want to use for training (e.g., title, plot, etc.).

2. Embedding Generation: The bot uses gte-small from the SentenceTransformer library to generate vector embeddings for text fields (e.g., title and plot) from your MongoDB collection.

3. Create Vector Index: The /trainbot endpoint creates a vector index in MongoDB for efficient vector search based on embeddings.

4. Query and Response Generation: When a query is made through the /response endpoint, the bot retrieves the most relevant documents from the MongoDB collection using the vector search. It then generates a response using
the Google gemma-2b-it language model based on the retrieved data.


How It Works:

1. Embedding Generation: For each document in the MongoDB collection, the bot combines specified fields (e.g., title, plot) and generates a vector embedding using gte-small.

2. Vector Search: When a user makes a query, the chatbot generates a vector embedding of the query and retrieves the most similar documents from MongoDB using vector search based on cosine similarity.

3. LLM Response Generation: The bot then passes the query and retrieved data into gemma-2b-it, which generates a coherent, context-aware response by combining both the query and the retrieved information.


License:

This project and all associated code are the exclusive property of the company and the owner. Unauthorized copying, modification, distribution, or use of this project or any of its components is strictly prohibited without explicit written permission from the owner.

For inquiries regarding usage permissions, please contact:

Hriday Sehgal

Email: hriday.career@gmail.com
