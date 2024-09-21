from sentence_transformers import SentenceTransformer
import pymongo
from huggingface_hub import InferenceClient, login
from typing import List
from key_param import MONGO_URI, HF_TOKEN 

# MongoDB URI
mongo_uri = MONGO_URI

# Function to connect to MongoDB
def get_mongo_client():
    """Establish connection to MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

# Model for embedding generation
embedding_model = SentenceTransformer("thenlper/gte-small")

# Authenticate Hugging Face only once when the server starts
login(token=HF_TOKEN, add_to_git_credential=True)

# Hugging Face Inference API for Google Gemma model
inference = InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN)

# Function to generate embeddings for a given text
def get_embedding(text: str) -> List[float]:
    """Generate embeddings for a given text."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# Function to train the model on MongoDB collection
def train_model_on_collection(data: List[dict], fields: List[str], collection):
    """Process the data, generate embeddings, and update MongoDB."""
    for item in data:
        combined_text = " ".join([str(item.get(field, "")) for field in fields])
        if not combined_text.strip():
            print(f"Invalid data encountered for document {item['_id']}, skipping...")
            continue

        # Generate embeddings for the combined fields
        embedding = get_embedding(combined_text)

        # Update the document in MongoDB with the generated embedding
        collection.update_one(
            {"_id": item["_id"]},  # Match by document ID
            {"$set": {"embedding": embedding}}  # Update with the embedding field
        )
    print("Data and embeddings have been successfully updated in MongoDB.")

# Function to create a vector index in MongoDB
def create_mongo_vector_index(collection, path: str, num_dimensions: int, similarity: str):
    try:
        # Define the vector index specification
        search_index_model = {
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": num_dimensions,
                        "path": path,  # The field that contains the vectors
                        "similarity": similarity  # Similarity metric
                    }
                ]
            },
            "name": f"{path}_vector_index",  # Name of the index
            "type": "vectorSearch"
        }
        
        # Create the vector index in MongoDB Atlas
        result = collection.create_search_index(search_index_model)
        print("Vector index created in MongoDB Atlas.")
        print(result)
    except Exception as e:
        print(f"Failed to create vector index: {e}")

# Function for vector search based on user query
def vector_search(user_query: str, collection):
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,
                "limit": 4
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "email": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    # Run the aggregation pipeline
    try:
        results = collection.aggregate(pipeline)
        return list(results) if results else []
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []

# Function to get the search result and format it for response
def get_search_result(query: str, collection):
    results = vector_search(query, collection)
    search_result = ""
    for result in results:
        search_result += f"Name: {result.get('name', 'N/A')}\nEmail: {result.get('email', 'N/A')}\n"
    return search_result

# Function to generate a response using Hugging Face Google Gemma API
def generate_response_with_gemma(query, collection):
    # Get search result from MongoDB
    source_information = get_search_result(query, collection)
    
    # Combine query and search results for context
    combined_information = f"Query: {query}\nHere are some search results. Based on these, answer the following query:\n{source_information}"

    # Prepare the messages for chat completion
    messages = [{"role": "user", "content": combined_information}]
    
    # Log the combined information
    print("Sending to Gemma Model:", combined_information)
    
    try:
        # Stream the response as it's generated
        response_stream = inference.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=True
        )

        # Collect and return the full response from the stream
        response_text = ""
        for message in response_stream:
            response_text += message.choices[0].delta.content

        print("Gemma Response:", response_text)
        return response_text
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return "An error occurred while generating the response."
