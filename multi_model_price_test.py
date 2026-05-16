import boto3
import time
import json
import re
import os
import csv
import math
from datetime import datetime
from statistics import mean, stdev
from typing import List, Dict, Tuple, Optional, Any
from botocore.exceptions import ClientError

# -----------------------------------------------------------------------------
# 1. AWS Clients (ap-southeast-2)
# -----------------------------------------------------------------------------
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="ap-southeast-2")
bedrock_management = boto3.client(service_name="bedrock", region_name="ap-southeast-2")

# -----------------------------------------------------------------------------
# 2. Pricing per 1,000 tokens (ap-southeast-2) – all models (conversational + embedding)
#    Sources: AWS Bedrock Pricing page (May 2026)
# -----------------------------------------------------------------------------
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Google Gemma
    "google.gemma-3-4b-it":   {"input": 0.0002, "output": 0.0006},
    "google.gemma-3-12b-it":  {"input": 0.0005, "output": 0.0015},
    "google.gemma-3-27b-it":  {"input": 0.0010, "output": 0.0030},
    # NVIDIA Nemotron
    "nvidia.nemotron-nano-12b-v2":   {"input": 0.0004, "output": 0.0012},
    "nvidia.nemotron-super-3-120b":  {"input": 0.0030, "output": 0.0090},
    "nvidia.nemotron-nano-3-30b":    {"input": 0.0005, "output": 0.0015},
    "nvidia.nemotron-nano-9b-v2":    {"input": 0.0003, "output": 0.0009},
    # Anthropic Claude
    "anthropic.claude-opus-4-7":                         {"input": 0.0150, "output": 0.0750},
    "anthropic.claude-haiku-4-5-20251001-v1:0":          {"input": 0.0008, "output": 0.0040},
    "anthropic.claude-sonnet-4-5-20250929-v1:0":         {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-opus-4-5-20251101-v1:0":           {"input": 0.0150, "output": 0.0750},
    "anthropic.claude-sonnet-4-6":                       {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-opus-4-6-v1":                      {"input": 0.0150, "output": 0.0750},
    "anthropic.claude-3-sonnet-20240229-v1:0:28k":       {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-3-sonnet-20240229-v1:0:200k":      {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-3-sonnet-20240229-v1:0":           {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-3-haiku-20240307-v1:0:48k":        {"input": 0.0008, "output": 0.0040},
    "anthropic.claude-3-haiku-20240307-v1:0:200k":       {"input": 0.0008, "output": 0.0040},
    "anthropic.claude-3-haiku-20240307-v1:0":            {"input": 0.0008, "output": 0.0040},
    "anthropic.claude-3-5-sonnet-20240620-v1:0":         {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-3-5-sonnet-20241022-v2:0":         {"input": 0.0030, "output": 0.0150},
    "anthropic.claude-sonnet-4-20250514-v1:0":           {"input": 0.0030, "output": 0.0150},
    # OpenAI GPT‑OSS
    "openai.gpt-oss-120b-1:0":  {"input": 0.0075, "output": 0.0300},
    "openai.gpt-oss-20b-1:0":   {"input": 0.0015, "output": 0.0060},
    # Amazon Nova
    "amazon.nova-2-lite-v1:0":  {"input": 0.0002, "output": 0.0006},
    "amazon.nova-pro-v1:0":     {"input": 0.0008, "output": 0.0024},
    "amazon.nova-lite-v1:0":    {"input": 0.0002, "output": 0.0006},
    "amazon.nova-micro-v1:0":   {"input": 0.0001, "output": 0.0003},
    # Mistral
    "mistral.mistral-large-3-675b-instruct":  {"input": 0.0080, "output": 0.0240},
    "mistral.magistral-small-2509":           {"input": 0.0010, "output": 0.0030},
    "mistral.ministral-3-3b-instruct":        {"input": 0.0008, "output": 0.0024},
    "mistral.ministral-3-14b-instruct":       {"input": 0.0012, "output": 0.0036},
    "mistral.ministral-3-8b-instruct":        {"input": 0.0010, "output": 0.0030},
    "mistral.voxtral-small-24b-2507":         {"input": 0.0015, "output": 0.0045},
    "mistral.mistral-7b-instruct-v0:2":       {"input": 0.0005, "output": 0.0015},
    "mistral.mixtral-8x7b-instruct-v0:1":     {"input": 0.0010, "output": 0.0030},
    "mistral.mistral-large-2402-v1:0":        {"input": 0.0080, "output": 0.0240},
    # Embedding models (input only, output tokens = 0)
    "cohere.embed-english-v3":          {"input": 0.0001, "output": 0.0},
    "amazon.titan-embed-text-v2:0":     {"input": 0.00002, "output": 0.0},
    "cohere.embed-multilingual-v3":     {"input": 0.0001, "output": 0.0},
}
# Default fallback (used if model not found – will print warning)
DEFAULT_PRICING = {"input": 0.0010, "output": 0.0030}

# Embedding models (special handling)
EMBEDDING_MODELS = {
    "cohere.embed-english-v3",
    "amazon.titan-embed-text-v2:0",
    "cohere.embed-multilingual-v3",
}
# List of embedding models to test (in addition to conversational models)
EMBEDDING_MODEL_IDS = list(EMBEDDING_MODELS)

# -----------------------------------------------------------------------------
# 3. Model Lists (conversational)
# -----------------------------------------------------------------------------
MODEL_IDS: List[str] = [
    "google.gemma-3-4b-it",
    "google.gemma-3-12b-it",
    "google.gemma-3-27b-it",
    "nvidia.nemotron-nano-12b-v2",
    "nvidia.nemotron-super-3-120b",
    "nvidia.nemotron-nano-3-30b",
    "nvidia.nemotron-nano-9b-v2",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-3-sonnet-20240229-v1:0:28k",
    "anthropic.claude-3-sonnet-20240229-v1:0:200k",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0:48k",
    "anthropic.claude-3-haiku-20240307-v1:0:200k",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b-1:0",
    "amazon.nova-2-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "mistral.mistral-large-3-675b-instruct",
    "mistral.magistral-small-2509",
    "mistral.ministral-3-3b-instruct",
    "mistral.ministral-3-14b-instruct",
    "mistral.ministral-3-8b-instruct",
    "mistral.voxtral-small-24b-2507",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

# -----------------------------------------------------------------------------
# 4. Test Data – 20 conversational queries with keywords
# -----------------------------------------------------------------------------
CONVERSATION_TEST_CASES = [
    {"query": "What is the capital of Australia?", "keywords": ["canberra"], "source": "https://www.australia.com/en/facts-and-planning/about-australia/cities/canberra.html"},
    {"query": "Who painted the Mona Lisa?", "keywords": ["leonardo", "da vinci"], "source": "https://www.louvre.fr/en/explore/the-palace/from-the-chateau-to-the-museum-mona-lisa"},
    {"query": "If a train travels at 60 km/h for 2 hours, how far does it go?", "keywords": ["120", "km"], "source": "https://www.mathsisfun.com/measure/speed-distance-time.html"},
    {"query": "Solve for x: 2x + 5 = 13", "keywords": ["x = 4", "x=4"], "source": "https://www.mathsisfun.com/algebra/equations-solving.html"},
    {"query": "What is compound interest?", "keywords": ["interest on interest", "interest on principal"], "source": "https://www.investopedia.com/terms/c/compoundinterest.asp"},
    {"query": "What does 'habeas corpus' mean?", "keywords": ["produce the body", "bring the body"], "source": "https://www.law.cornell.edu/wex/habeas_corpus"},
    {"query": "Explain the theory of relativity in simple terms.", "keywords": ["einstein", "space", "time", "gravity"], "source": "https://www.space.com/17661-theory-general-relativity.html"},
    {"query": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "keywords": ["no", "cannot", "not necessarily"], "source": "https://www.logicmountain.com/fallacies/quantifier-fallacy"},
    {"query": "When did World War II end?", "keywords": ["1945"], "source": "https://www.history.com/topics/world-war-ii/world-war-ii-history"},
    {"query": "What is photosynthesis?", "keywords": ["plants", "sunlight", "energy", "carbon dioxide", "oxygen"], "source": "https://www.nationalgeographic.org/encyclopedia/photosynthesis/"},
    {"query": "What is inflation?", "keywords": ["general increase", "prices", "purchasing power"], "source": "https://www.imf.org/en/Publications/fandd/issues/Series/Back-to-Basics/Inflation"},
    {"query": "What is a transformer in machine learning?", "keywords": ["attention", "sequence", "neural network"], "source": "https://arxiv.org/abs/1706.03762"},
    {"query": "What is the function of the human liver?", "keywords": ["detoxification", "protein synthesis", "bile"], "source": "https://www.britannica.com/science/liver"},
    {"query": "Which river is the longest in the world?", "keywords": ["nile"], "source": "https://www.britannica.com/story/is-the-nile-the-longest-river-in-the-world"},
    {"query": "Who was the first President of the United States?", "keywords": ["george washington"], "source": "https://www.whitehouse.gov/about-the-white-house/presidents/george-washington/"},
    {"query": "Who wrote 'Pride and Prejudice'?", "keywords": ["jane austen"], "source": "https://www.britannica.com/biography/Jane-Austen"},
    {"query": "What is the 'categorical imperative'?", "keywords": ["kant", "universal law", "duty", "moral"], "source": "https://plato.stanford.edu/entries/kant-moral/"},
    {"query": "Who won the FIFA World Cup in 2022?", "keywords": ["argentina"], "source": "https://www.fifa.com/tournaments/mens/worldcup/qatar2022"},
    {"query": "What is 'due process'?", "keywords": ["fair treatment", "judicial", "legal proceedings"], "source": "https://www.law.cornell.edu/wex/due_process"},
    {"query": "What is the probability of rolling a 6 on a fair die?", "keywords": ["1/6", "one sixth", "0.1667"], "source": "https://www.mathsisfun.com/data/probability.html"},
]

# -----------------------------------------------------------------------------
# 5. Embedding Test Pairs (20 pairs: 10 similar, 5 dissimilar, 5 negation)
#    Each pair: (sentence_a, sentence_b, expected_similar: True/False)
# -----------------------------------------------------------------------------
EMBEDDING_TEST_PAIRS = [
    # Similar pairs (semantic similarity)
    ("A cat sleeps on the mat.", "A feline naps on the rug.", True),
    ("Machine learning models require large amounts of data.", "Deep learning algorithms need extensive datasets.", True),
    ("The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps above a sleepy hound.", True),
    ("She loves to travel to new places.", "She enjoys visiting unfamiliar destinations.", True),
    ("Python is a popular programming language.", "Python is widely used for coding.", True),
    ("The stock market experienced a downturn.", "Equities saw a significant decline.", True),
    ("Climate change is a pressing global issue.", "Global warming is an urgent worldwide problem.", True),
    ("The painting used vibrant colors and bold brushstrokes.", "The artwork featured bright hues and strong strokes.", True),
    ("Yoga and meditation help reduce stress.", "Practicing mindfulness lowers anxiety.", True),
    ("The athlete broke the world record.", "The runner shattered the global benchmark.", True),
    
    # Dissimilar pairs (no negation)
    ("The sun rises in the east.", "The restaurant serves delicious pasta.", False),
    ("How to bake a chocolate cake.", "Quantum physics explains particle behavior.", False),
    ("The Great Wall of China is ancient.", "The new smartphone has a great camera.", False),
    ("Space exploration leads to innovation.", "The movie had a thrilling adventure plot.", False),
    ("Blockchain enables secure transactions.", "The novel explores themes of identity.", False),
    
    # Negation pairs (expected dissimilar)
    ("The sky is blue.", "The sky is not blue.", False),
    ("Water boils at 100 degrees Celsius.", "Water does not boil at 100 degrees Celsius.", False),
    ("Cats are mammals.", "Cats are not mammals.", False),
    ("The Earth orbits the Sun.", "The Earth does not orbit the Sun.", False),
    ("Snow is white.", "Snow is not white.", False),
]

# -----------------------------------------------------------------------------
# 6. Test Configuration
# -----------------------------------------------------------------------------
REQUEST_DELAY = 1.0      # seconds between requests per model
MODEL_DELAY = 5.0        # seconds between different models
MAX_RETRIES = 3
COSINE_THRESHOLD = 0.5   # threshold for treating two embeddings as "similar"

# -----------------------------------------------------------------------------
# 7. Helper Functions
# -----------------------------------------------------------------------------
def get_model_parameter_size(model_id: str) -> str:
    """Extract parameter size from model ID (e.g., '120b' → '120B')."""
    match = re.search(r'(\d+(?:\.\d+)?)b', model_id, re.IGNORECASE)
    if match:
        return f"{match.group(1)}B"
    return "N/A"

def get_model_pricing(model_id: str) -> Dict[str, float]:
    """Return input/output pricing per 1K tokens; warns if using default."""
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    print(f"Warning: No pricing found for {model_id}, using default (${DEFAULT_PRICING['input']}/1K in, ${DEFAULT_PRICING['output']}/1K out)")
    return DEFAULT_PRICING

def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> Tuple[float, float, float]:
    """Return (total_cost, input_cost, output_cost) in USD."""
    pricing = get_model_pricing(model_id)
    in_cost = (input_tokens / 1000) * pricing["input"]
    out_cost = (output_tokens / 1000) * pricing["output"]
    return in_cost + out_cost, in_cost, out_cost

def check_model_access(model_id: str) -> Tuple[bool, Optional[str]]:
    """Check if model is accessible in the current region."""
    try:
        if model_id in EMBEDDING_MODELS:
            if "cohere" in model_id:
                body = json.dumps({"texts": ["test"], "input_type": "search_query"})
            else:
                body = json.dumps({"inputText": "test"})
            bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )
        else:
            bedrock_runtime.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.5}
            )
        return True, None
    except ClientError as e:
        return False, f"{e.response['Error']['Code']}: {e.response['Error']['Message']}"
    except Exception as e:
        return False, str(e)

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def get_embedding(text: str, model_id: str):
    """
    Return embedding vector, latency, input_tokens, success, cost.
    """
    start = time.perf_counter()
    try:
        if "cohere" in model_id:
            body = json.dumps({"texts": [text], "input_type": "search_query"})
        else:
            body = json.dumps({"inputText": text})
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        latency = time.perf_counter() - start
        resp_body = json.loads(response['body'].read())
        # Extract vector
        if "cohere" in model_id:
            embedding = resp_body.get('embeddings', [])[0] if resp_body.get('embeddings') else None
        else:
            embedding = resp_body.get('embedding')
        # Extract input token count
        if "amazon.titan" in model_id:
            in_tokens = resp_body.get('inputTextTokenCount', 0)
        elif "cohere" in model_id:
            in_tokens = resp_body.get('meta', {}).get('billed_units', {}).get('input_tokens', 0)
        else:
            in_tokens = 0
        out_tokens = 0
        cost, in_cost, out_cost = calculate_cost(in_tokens, out_tokens, model_id)
        return {
            "success": True,
            "embedding": embedding,
            "latency_sec": latency,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_cost": cost,
            "input_cost": in_cost,
            "output_cost": out_cost,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "embedding": None,
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0,
            "input_cost": 0,
            "output_cost": 0,
            "error": str(e)
        }

def is_correct(model_answer: str, keywords: List[str]) -> bool:
    """Check if conversational answer contains all keywords."""
    if not model_answer:
        return False
    cleaned = re.sub(r'[^\w\s]', '', model_answer.lower())
    words = set(cleaned.split())
    for kw in keywords:
        kw_lower = kw.lower()
        if ' ' in kw_lower:
            if kw_lower not in cleaned:
                return False
        else:
            if kw_lower not in words:
                return False
    return True

def invoke_conversational(prompt: str, model_id: str):
    """Send a conversational prompt, return result dict."""
    start = time.perf_counter()
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.5}
        )
        latency = time.perf_counter() - start
        usage = response.get('usage', {})
        in_tokens = usage.get('inputTokens', 0)
        out_tokens = usage.get('outputTokens', 0)
        content = response.get('output', {}).get('message', {}).get('content', [])
        answer_text = content[0].get('text', '') if content else ''
        total_cost, in_cost, out_cost = calculate_cost(in_tokens, out_tokens, model_id)
        return {
            "success": True,
            "latency_sec": latency,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_cost": total_cost,
            "input_cost": in_cost,
            "output_cost": out_cost,
            "answer_text": answer_text,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0,
            "input_cost": 0,
            "output_cost": 0,
            "answer_text": "",
            "error": str(e)
        }

# -----------------------------------------------------------------------------
# 8. Test Suites
# -----------------------------------------------------------------------------
def run_conversational_tests(model_id: str):
    """Run 20 conversational test cases, return results, stats, total_cost."""
    correct = 0
    results = []
    total_cost = 0.0
    print(f"\nTesting Conversational Model: {model_id}")
    pricing = get_model_pricing(model_id)
    print(f"Pricing: ${pricing['input']}/1K in, ${pricing['output']}/1K out")
    accessible, err = check_model_access(model_id)
    if not accessible:
        print(f"Not accessible: {err}")
        return results, 0, 0.0
    print("Accessible")
    for idx, test in enumerate(CONVERSATION_TEST_CASES, 1):
        print(f"Query {idx}/{len(CONVERSATION_TEST_CASES)}: {test['query'][:50]}...", end="\r")
        attempt = 0
        res = None
        while attempt < MAX_RETRIES:
            res = invoke_conversational(test['query'], model_id)
            if res['success']:
                break
            if "Throttling" in str(res.get('error', '')):
                wait = (2 ** attempt) * 2
                print(f"\nThrottled, waiting {wait}s...")
                time.sleep(wait)
                attempt += 1
            else:
                break
        if res and res['success']:
            if is_correct(res['answer_text'], test['keywords']):
                correct += 1
            results.append(res)
            total_cost += res['total_cost']
        else:
            results.append({
                "success": False,
                "query": test['query'],
                "error": res.get('error', 'Unknown') if res else "No response",
                "total_cost": 0
            })
        time.sleep(REQUEST_DELAY)
    print(f"\nCompleted. Correct: {correct}/{len(CONVERSATION_TEST_CASES)}")
    return results, correct, total_cost

def run_embedding_tests(model_id: str, test_pairs):
    """
    Run embedding similarity test.
    Returns: (list of detailed results, precision, recall, f1, total_cost)
    """
    results = []
    tp = fp = tn = fn = 0
    total_cost = 0.0
    print(f"\nTesting Embedding Model: {model_id}")
    pricing = get_model_pricing(model_id)
    print(f"   Pricing: ${pricing['input']}/1K input (output free)")
    accessible, err = check_model_access(model_id)
    if not accessible:
        print(f"Not accessible: {err}")
        return results, 0.0, 0.0, 0.0, 0.0
    print("Accessible")
    for idx, (sent_a, sent_b, expected_similar) in enumerate(test_pairs, 1):
        print(f"   Pair {idx}/{len(test_pairs)}: {sent_a[:40]}... vs {sent_b[:40]}...", end="\r")
        # Get embedding A
        attempt_a = 0
        res_a = None
        while attempt_a < MAX_RETRIES:
            res_a = get_embedding(sent_a, model_id)
            if res_a['success']:
                break
            if "Throttling" in str(res_a.get('error', '')):
                wait = (2 ** attempt_a) * 2
                print(f"\nThrottled, waiting {wait}s...")
                time.sleep(wait)
                attempt_a += 1
            else:
                break
        # Get embedding B
        attempt_b = 0
        res_b = None
        while attempt_b < MAX_RETRIES:
            res_b = get_embedding(sent_b, model_id)
            if res_b['success']:
                break
            if "Throttling" in str(res_b.get('error', '')):
                wait = (2 ** attempt_b) * 2
                print(f"\nThrottled, waiting {wait}s...")
                time.sleep(wait)
                attempt_b += 1
            else:
                break
        if res_a and res_a['success'] and res_b and res_b['success']:
            sim = cosine_similarity(res_a['embedding'], res_b['embedding'])
            predicted_similar = sim > COSINE_THRESHOLD
            correct = (predicted_similar == expected_similar)
            # Update confusion matrix
            if predicted_similar and expected_similar:
                tp += 1
            elif predicted_similar and not expected_similar:
                fp += 1
            elif not predicted_similar and not expected_similar:
                tn += 1
            elif not predicted_similar and expected_similar:
                fn += 1
            # Aggregate cost (both calls)
            pair_cost = res_a['total_cost'] + res_b['total_cost']
            total_cost += pair_cost
            results.append({
                "success": True,
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "expected_similar": expected_similar,
                "cosine_sim": sim,
                "predicted_similar": predicted_similar,
                "correct": correct,
                "latency_a": res_a['latency_sec'],
                "latency_b": res_b['latency_sec'],
                "tokens_a": res_a['input_tokens'],
                "tokens_b": res_b['input_tokens'],
                "cost_a": res_a['total_cost'],
                "cost_b": res_b['total_cost']
            })
        else:
            results.append({
                "success": False,
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "error_a": res_a.get('error') if res_a else "No response",
                "error_b": res_b.get('error') if res_b else "No response"
            })
        time.sleep(REQUEST_DELAY)
    # Compute precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\nCompleted. Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return results, precision, recall, f1, total_cost

# -----------------------------------------------------------------------------
# 9. Aggregation and Output
# -----------------------------------------------------------------------------
def aggregate_conversational_stats(results, correct_count, total_queries):
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        return {
            "success_rate": 0.0,
            "correct_rate": (correct_count / total_queries) * 100,
            "correct_count": correct_count,
            "avg_latency": 0.0,
            "min_latency": 0.0,
            "max_latency": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "total_cost": 0.0,
            "avg_cost_per_req": 0.0,
            "cost_per_1k_tokens": 0.0,
            "tokens_per_second": 0.0,
            "total_tokens": 0
        }
    latencies = [r['latency_sec'] for r in successful]
    in_tokens = [r['input_tokens'] for r in successful]
    out_tokens = [r['output_tokens'] for r in successful]
    costs = [r['total_cost'] for r in successful]
    total_time = sum(latencies)
    total_out = sum(out_tokens)
    total_in = sum(in_tokens)
    total_tokens = total_in + total_out
    tps = total_out / total_time if total_time > 0 else 0
    total_cost = sum(costs)
    cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
    return {
        "success_rate": len(successful) / len(results) * 100,
        "correct_rate": (correct_count / total_queries) * 100,
        "correct_count": correct_count,
        "avg_latency": mean(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "avg_input_tokens": mean(in_tokens),
        "avg_output_tokens": mean(out_tokens),
        "total_cost": total_cost,
        "avg_cost_per_req": mean(costs),
        "cost_per_1k_tokens": cost_per_1k,
        "tokens_per_second": tps,
        "total_tokens": total_tokens
    }

def print_conversational_summary(model_id, stats):
    print(f"\n{'='*50}")
    print(f"CONVERSATIONAL MODEL: {model_id}")
    print(f"  Parameter size: {get_model_parameter_size(model_id)}")
    print(f"  Success rate:   {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total']})")
    print(f"  Correct rate:   {stats['correct_rate']:.1f}% ({stats['correct_count']}/{stats['total']})")
    print(f"  Avg latency:    {stats['avg_latency']:.4f} s")
    print(f"  Min latency:    {stats['min_latency']:.4f} s")
    print(f"  Max latency:    {stats['max_latency']:.4f} s")
    print(f"  Total cost:     ${stats['total_cost']:.6f}")
    print(f"{'='*50}")

def print_embedding_summary(model_id, precision, recall, f1, total_cost, total_pairs):
    print(f"\n{'='*50}")
    print(f"EMBEDDING MODEL: {model_id}")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1 Score:       {f1:.3f}")
    print(f"  Total cost:     ${total_cost:.6f} (for {total_pairs} pairs)")
    print(f"{'='*50}")

def print_combined_summary(conv_results, embed_results):
    """Print a combined table of all models (conversational + embedding) with cost metrics."""
    print("\n" + "="*130)
    print("FINAL SUMMARY: COST & PERFORMANCE (ap-southeast-2)")
    print("="*130)
    
    # Conversational header
    print(f"{'Type':<10} {'Model ID':<42} {'Succ Rate':<10} {'Correct Rate':<10} {'Avg Lat (s)':<10} {'Cost/Req':<12} {'Cost/1K':<10} {'Total Cost':<12}")
    print("-"*130)
    for model_id, stats in conv_results.items():
        short_name = model_id if len(model_id) <= 42 else model_id[:39] + "..."
        succ_rate = f"{stats['successful']}/{stats['total']}"
        correct_rate = f"{stats['correct_count']}/{stats['total']} ({stats['correct_rate']:.1f}%)"
        print(f"{'Conv':<10} {short_name:<42} {succ_rate:<10} {correct_rate:<10} {stats['avg_latency']:<10.4f} ${stats['avg_cost_per_req']:<11.6f} ${stats['cost_per_1k_tokens']:<9.6f} ${stats['total_cost']:<11.6f}")
    
    # Embedding section
    print("-"*130)
    print(f"{'Type':<10} {'Model ID':<42} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Total Cost':<12}")
    print("-"*130)
    for model_id, stats in embed_results.items():
        short_name = model_id if len(model_id) <= 42 else model_id[:39] + "..."
        print(f"{'Embed':<10} {short_name:<42} {stats['precision']:<10.3f} {stats['recall']:<10.3f} {stats['f1']:<10.3f} ${stats['total_cost']:<11.6f}")
    print("="*130)

def save_all_results(conv_results, embed_results, conv_raw, embed_raw):
    """Save all results (conversational + embedding) to CSV in ../experiment_results/"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(parent_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_model_evaluation_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model Type", "Model ID", "Param Size", "Success Rate", "Correct Rate / F1",
                         "Avg Latency (s)", "Min Latency (s)", "Max Latency (s)",
                         "Total Input Tokens", "Total Output Tokens", "Total Cost (USD)"])
        # Conversational summaries
        for model_id, stats in conv_results.items():
            writer.writerow([
                "Conversational", model_id, stats.get("param_size", "N/A"),
                f"{stats['success_rate']:.1f}%", f"{stats['correct_rate']:.1f}%",
                f"{stats['avg_latency']:.4f}", f"{stats['min_latency']:.4f}", f"{stats['max_latency']:.4f}",
                stats.get("total_input_tokens", 0), stats.get("total_output_tokens", 0),
                f"{stats['total_cost']:.6f}"
            ])
        # Embedding summaries
        for model_id, stats in embed_results.items():
            writer.writerow([
                "Embedding", model_id, "N/A",
                "N/A", f"F1={stats['f1']:.3f}",
                "N/A", "N/A", "N/A",
                "N/A", "N/A", f"{stats['total_cost']:.6f}"
            ])
        writer.writerow([])
        writer.writerow(["DETAILED PER-REQUEST (Conversational)"])
        writer.writerow(["Model ID", "Query", "Success", "Latency (s)", "Input Tokens", "Output Tokens",
                         "Total Cost", "Correct"])
        for model_id, reqs in conv_raw.items():
            for r in reqs:
                writer.writerow([
                    model_id, r.get('query', '')[:100], r.get('success', False),
                    round(r.get('latency_sec', 0), 4), r.get('input_tokens', 0), r.get('output_tokens', 0),
                    round(r.get('total_cost', 0), 8), r.get('correct', False)
                ])
        writer.writerow([])
        writer.writerow(["DETAILED PER-PAIR (Embedding)"])
        writer.writerow(["Model ID", "Sentence A", "Sentence B", "Expected", "Cosine Sim", "Predicted",
                         "Correct", "Latency A", "Latency B", "Tokens A", "Tokens B", "Cost A", "Cost B"])
        for model_id, pairs in embed_raw.items():
            for p in pairs:
                if p.get('success'):
                    writer.writerow([
                        model_id, p['sentence_a'][:80], p['sentence_b'][:80], p['expected_similar'],
                        f"{p['cosine_sim']:.4f}", p['predicted_similar'], p['correct'],
                        f"{p['latency_a']:.4f}", f"{p['latency_b']:.4f}",
                        p['tokens_a'], p['tokens_b'], f"{p['cost_a']:.8f}", f"{p['cost_b']:.8f}"
                    ])
                else:
                    writer.writerow([model_id, p['sentence_a'][:80], p['sentence_b'][:80], "ERROR", "", "", "", "", "", "", "", "", ""])
    print(f"\nFull results saved to: {filepath}")

# -----------------------------------------------------------------------------
# 10. Main Execution
# -----------------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("AWS BEDROCK MULTI-MODEL EVALUATION (Conversational + Embedding)")
    print("Region: ap-southeast-2")
    print("="*70)
    
    # 1. Test conversational models
    print("\n" + "="*70)
    print("PHASE 1: CONVERSATIONAL MODELS")
    print("="*70)
    conv_results = {}
    conv_raw_data = {}
    for model_id in MODEL_IDS:
        results, correct, total_cost = run_conversational_tests(model_id)
        conv_raw_data[model_id] = results
        stats = aggregate_conversational_stats(results, correct, len(CONVERSATION_TEST_CASES))
        stats["param_size"] = get_model_parameter_size(model_id)
        stats["total"] = len(CONVERSATION_TEST_CASES)
        stats["successful"] = len([r for r in results if r.get('success')])
        conv_results[model_id] = stats
        print_conversational_summary(model_id, stats)
        if model_id != MODEL_IDS[-1]:
            print(f"\nWaiting {MODEL_DELAY}s before next model...")
            time.sleep(MODEL_DELAY)
    
    # 2. Test embedding models
    print("\n" + "="*70)
    print("PHASE 2: EMBEDDING MODELS (Semantic Similarity)")
    print("="*70)
    embed_results = {}
    embed_raw_data = {}
    for model_id in EMBEDDING_MODEL_IDS:
        results, prec, rec, f1, total_cost = run_embedding_tests(model_id, EMBEDDING_TEST_PAIRS)
        embed_raw_data[model_id] = results
        embed_results[model_id] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "total_cost": total_cost,
            "total_pairs": len(EMBEDDING_TEST_PAIRS)
        }
        print_embedding_summary(model_id, prec, rec, f1, total_cost, len(EMBEDDING_TEST_PAIRS))
        if model_id != EMBEDDING_MODEL_IDS[-1]:
            print(f"\nWaiting {MODEL_DELAY}s before next model...")
            time.sleep(MODEL_DELAY)
    
    # 3. Print combined summary table
    print_combined_summary(conv_results, embed_results)
    
    # 4. Save everything
    save_all_results(conv_results, embed_results, conv_raw_data, embed_raw_data)
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()