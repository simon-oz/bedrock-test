"""
Access AWS managed service via boto3, an AWS SDK for Python,
which allows developers to write software that makes use of services like Amazon S3 and Amazon EC2
"""

import boto3
import json

# create a Bedrock Runtime client in the AWS Region ap-southeast-2
client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")

# Set the mode ID
model_id = "amazon.titan-embed-text-v2:0"

# The text to convert to an embedding
input_text1 = "who is the USA current presendent?"
input_text2 = "who is the USA current presendent"

# Create the request for the model
native_request1 = {"inputText": input_text1}
native_request2 = {"inputText": input_text2}

# Convert the native request to JSON
request1 = json.dumps(native_request1)
request2 = json.dumps(native_request2)

# Invoke the model with the request
response1 = client.invoke_model(modelId=model_id, body=request1)
response2 = client.invoke_model(modelId=model_id, body=request2)

# Decode the model's native response body
model_response1 = json.loads(response1["body"].read())
model_response2 = json.loads(response2["body"].read())


# Extract and print the generated embedding and the input text token count.
embedding1 = model_response1["embedding"]
embedding2 = model_response2["embedding"]

input_token_count1 = model_response1["inputTextTokenCount"]
input_token_count2 = model_response2["inputTextTokenCount"]

print("\nInput str1: ", input_text1)
print(f"# of input tokens: {input_token_count1}")
print(f"Size of generated embeddings: {len(embedding1)}")
print("Embedding")
print(embedding1)

print("\nInput str2: ", input_text2)
print(f"# of input tokens: {input_token_count2}")
print(f"Size of generated embeddings: {len(embedding2)}")
print("Embedding")
print(embedding2)




