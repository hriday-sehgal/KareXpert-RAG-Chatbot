from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pymongo
from  bot import get_mongo_client, train_model_on_collection, create_mongo_vector_index, generate_response_with_gemma, vector_search
import asyncio


# Initialize FastAPI app
app = FastAPI()


# Define input model for training and vector index creation
class TrainBotRequest(BaseModel):
    collection_name: str  # Name of the collection
    fields: List[str]  # Fields to use for embedding generation (e.g., title, plot, fullplot)
    path: str  # Path to create index on (e.g., "embedding")
    num_dimensions: int  # Number of dimensions in the embedding
    similarity: str  # Similarity function (cosine, dotProduct, etc.)

# Define input model for Response API
class ResponseRequest(BaseModel):
    collection_name: str  # Collection to search
    query: str  # User query

# Function to dynamically select fields from the collection
def get_dynamic_fields(collection):
    # Retrieve a sample document from the collection
    sample_doc = collection.find_one()
    if sample_doc:
        return list(sample_doc.keys())
    else:
        raise HTTPException(status_code=404, detail="No data found in the collection to determine fields")

# Endpoint to train the model, generate embeddings, and create a vector index
@app.post("/trainbot")
async def train_bot(request: TrainBotRequest):
    try:
        # Get MongoDB client
        mongo_client = get_mongo_client()
        if not mongo_client:
            raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")
        
        # Get the specified collection
        db = mongo_client["sample_mflix"]
        collection = db[request.collection_name]

        # Fetch the available fields from the collection dynamically
        available_fields = get_dynamic_fields(collection)
        
        # Ensure the provided fields exist in the collection
        invalid_fields = [field for field in request.fields if field not in available_fields]
        if invalid_fields:
            raise HTTPException(status_code=400, detail=f"Invalid fields: {invalid_fields}")

        # Fetch the data based on the requested fields
        projection = {field: 1 for field in request.fields}  # Project only the specified fields
        data = list(collection.find({}, projection))

        if not data:
            raise HTTPException(status_code=404, detail="No data found in the collection")
        
        # Call the function to process the data and generate embeddings
        train_model_on_collection(data, request.fields, collection)

        # Create a vector index in MongoDB Atlas
        create_mongo_vector_index(collection, request.path, request.num_dimensions, request.similarity)
        
        return {"message": "Model trained successfully, embeddings generated, and vector index created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training and index creation failed: {str(e)}")



# Endpoint to handle user query and generate both MongoDB and Gemma model responses
@app.post("/response")
async def get_response(request: ResponseRequest):
    try:
        mongo_client = get_mongo_client()
        if not mongo_client:
            raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")
        
        db = mongo_client["sample_mflix"]
        collection = db[request.collection_name]
        available_fields = get_dynamic_fields(collection)
        mongo_vector_response = vector_search(request.query, collection)
        
        if not mongo_vector_response:
            raise HTTPException(status_code=404, detail="No MongoDB results found for the query")

        formatted_mongo_response = "MongoDB Vector Search Results:\n"
        for result in mongo_vector_response:
            formatted_mongo_response += "\n".join([f"{field.capitalize()}: {result.get(field, 'N/A')}" for field in available_fields if field in result]) + "\n\n"

        # Run Gemma model on a separate thread
        response = await asyncio.to_thread(generate_response_with_gemma, request.query, collection)
        return {
            "mongo_response": formatted_mongo_response,
            "gemma_response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

