"""
Access AWS managed service via boto3, an AWS SDK for Python,
which allows to write software that makes use of services from Amazon EC2
"""

import boto3
import json

# create a Bedrock Runtime client in the AWS Region ap-southeast-2
client = boto3.client(service_name="bedrock-runtime", region_name="ap-southeast-2")


import boto3
import re

def get_model_response(input_text, model_id):

    # Build request body based on model type
    if "cohere" in model_id:
        # Cohere models: texts array + input_type
        native_request = {
            "texts": [input_text],          # Note: Cohere expects a list
            "input_type": "search_document"  # Typical use case for indexing
        }
    elif "titan" in model_id:
        # Titan models: inputText string
        native_request = {"inputText": input_text}
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    request = json.dumps(native_request)

    # Invoke the model with the request
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body
    model_response = json.loads(response["body"].read())


    # Extract and print the generated embedding and the input text token count.
    # Parse response based on model type
    if "cohere" in model_id:
        # Cohere returns "embeddings" as a list of lists
        embedding = model_response["embeddings"][0]   # First (only) embedding
        input_token_count = "N/A (not returned by Cohere)"
    elif "titan" in model_id:
        embedding = model_response["embedding"]
        input_token_count = model_response.get("inputTextTokenCount", "N/A")

    print("\n" + "="*50)
    print(f"Input string: {input_text}")
    print(f"Embedding model: {model_id}")
    print(f"Dimension of embeddings: {len(embedding)}")
    print(f"# of input tokens: {input_token_count}")
    print("Embedding (first 10 values):", embedding[:10])
    print("="*50)


def test_embed_model(model_id = "amazon.titan-embed-text-v2:0"):
    # Set the mode ID
    model_ids = [
        "cohere.embed-english-v3",
        "amazon.titan-embed-text-v2:0",
        "cohere.embed-multilingual-v3",
    ]

    # The text to convert to an embedding
    input_text = "Who is the USA current presendent?"

    # Convert the native request to JSON
    for model_id in model_ids:
        get_model_response(input_text, model_id)

if __name__ == "__main__":
    test_embed_model()


