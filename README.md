
# AWS Bedrock Model Evaluation Suite

This repository contains a set of Python scripts to test and evaluate various foundation models available through **AWS Bedrock** in the `ap-southeast-2` (Sydney) region. The suite covers:

- **Latency & token usage** for conversational and embedding models (`latency_test.py`)
- **Latency, cost estimation, and correctness** for conversational models + **semantic similarity (precision/recall/F1)** for embedding models (`multi_model_price_test.py`)
- **Safety guardrails evaluation** (spam, prompt injection, harmful content, multilingual, etc.) for OpenAI GPT‑OSS safeguard models (`gpt_oss_safeguard_test.py`)
- **Basic embedding extraction** example (`boto_test.py`)

All results are automatically saved as CSV files in the `experiment_results/` folder (created one level above the script directory).

---

## Prerequisites

- AWS account with Bedrock access enabled for the `ap-southeast-2` region.
- Appropriate IAM permissions (`bedrock:InvokeModel`, `bedrock:ListFoundationModels`, etc.)
- Python 3.9 or higher.
- Install dependencies from `requirements.txt`.

### Installation

```bash
# Clone or download the scripts into a directory, e.g., boto3-test/
cd boto3-test

# Install required packages
pip install -r requirements.txt

## Script Overview
1. boto_test.py – Basic Embedding Example
A minimal script to demonstrate how to call Bedrock embedding models (cohere.embed-english-v3, amazon.titan-embed-text-v2:0, cohere.embed-multilingual-v3) and print the resulting embedding vector (first 10 values) and input token count.

### Usage:

```bash
python boto_test.py
Output: Prints the embedding dimension, token count, and first 10 values for each embedding model.

## 2. latency_test.py – Latency & Token Usage (No Cost)
Tests all conversational models (39 models) on 20 factual and reasoning questions, and three embedding models on 20 sentence‑pair similarity tasks.
### Metrics measured:

- Latency (min, max, avg)

- Token counts (input/output for conversational; only input for embeddings)

- Success rate

- For conversational models: correctness (keyword‑based)

- For embedding models: semantic similarity accuracy (cosine similarity vs. ground truth labels)

### Outputs:

- Console summary per model

- CSV file: ./experiment_results/bedrock_latency_evaluation_<timestamp>.csv

### Run:

```bash
python latency_test.py

## 3. multi_model_price_test.py – Full Evaluation with Cost Estimation
Combines conversational and embedding tests with AWS Bedrock pricing (ap-southeast-2).

- Conversational: accuracy, latency, cost per request, cost per 1K tokens.

- Embedding: precision, recall, F1 score (using the same 20 sentence pairs, threshold 0.5), and total cost.

### Outputs:

Console summary with final comparison table

Detailed CSV: ./experiment_results/multi_model_evaluation_<timestamp>.csv
(includes per‑request conversational data and per‑pair embedding data)

### Run:

```bash
python multi_model_price_test.py
Note: Prices are estimates based on AWS public pricing as of May 2026. Verify on the AWS Bedrock Pricing page.

## 4. gpt_oss_safeguard_test.py – Safety Guardrail Evaluation
Evaluates openai.gpt-oss-safeguard-120b and -20b models on 10+ risk categories (spam, prompt injection, harmful instructions, jailbreak, self‑harm, fraud, multilingual harm, etc.).
### Uses the correct policy for each category (spam policy, injection policy, harm policy).
Reports per‑category accuracy, overall accuracy, adversarial refusal rate, reasoning effort impact, and performance.

### Outputs:

- Console summary with per‑category table

- CSV: ./experiment_results/gpt_oss_safeguard_evaluation_<timestamp>.csv

### Run:

```bash
python gpt_oss_safeguard_test.py

## Important Notes
All scripts use the region ap-southeast-2 (Sydney). Ensure your models are enabled there.

If a model is not accessible (not requested or not available), the script skips it and logs an error.

Throttling handling: exponential backoff with up to 3 retries.

Embedding similarity uses a fixed threshold of 0.5 (cosine). You can change this inside the scripts (COSINE_THRESHOLD variable).

Correctness for conversational models is based on keyword presence (case‑insensitive, punctuation ignored). This is a simple but effective measure.