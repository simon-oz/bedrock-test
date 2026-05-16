import boto3
import time
import json
import re
import os
import csv
import math
from datetime import datetime
from botocore.exceptions import ClientError
from statistics import mean, stdev

# ------------------------------------------------------------------------------
# 1. Setup Clients (ap-southeast-2)
# ------------------------------------------------------------------------------
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="ap-southeast-2")
bedrock_management = boto3.client(service_name="bedrock", region_name="ap-southeast-2")

# ------------------------------------------------------------------------------
# 2. Model Lists
# ------------------------------------------------------------------------------
# Conversational models (original list)
CONVERSATION_MODEL_IDS = [
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

# Embedding models (available in ap-southeast-2)
EMBEDDING_MODEL_IDS = [
    "cohere.embed-english-v3",
    "amazon.titan-embed-text-v2:0",
    "cohere.embed-multilingual-v3",
]

# ------------------------------------------------------------------------------
# 3. Test Configuration
# ------------------------------------------------------------------------------
REQUEST_DELAY = 1.0      # Seconds between requests
MAX_RETRIES = 3          # Max retries on throttling

# ------------------------------------------------------------------------------
# 4. Test Data
# ------------------------------------------------------------------------------
# 20 conversational test cases (with keywords for correctness)
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

# 20 embedding test pairs (semantic similarity evaluation)
# Each pair: (sentence_a, sentence_b, expected_similar: True/False)
# Includes 5 negation pairs (where B is the logical negation of A)
EMBEDDING_TEST_PAIRS = [
    # Similar pairs
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
    
    # Dissimilar pairs
    ("The sun rises in the east.", "The restaurant serves delicious pasta.", False),
    ("How to bake a chocolate cake.", "Quantum physics explains particle behavior.", False),
    ("The Great Wall of China is ancient.", "The new smartphone has a great camera.", False),
    ("Space exploration leads to innovation.", "The movie had a thrilling adventure plot.", False),
    ("Blockchain enables secure transactions.", "The novel explores themes of identity.", False),
    
    # Negation pairs (semantically dissimilar because one negates the other)
    ("The sky is blue.", "The sky is not blue.", False),
    ("Water boils at 100 degrees Celsius.", "Water does not boil at 100 degrees Celsius.", False),
    ("Cats are mammals.", "Cats are not mammals.", False),
    ("The Earth orbits the Sun.", "The Earth does not orbit the Sun.", False),
    ("Snow is white.", "Snow is not white.", False),
]

# ------------------------------------------------------------------------------
# 5. Helper Functions
# ------------------------------------------------------------------------------
def get_model_parameter_size(model_id: str) -> str:
    """Extract parameter size from model ID (e.g., '120b' → '120B')."""
    match = re.search(r'(\d+(?:\.\d+)?)b', model_id, re.IGNORECASE)
    if match:
        return f"{match.group(1)}B"
    return "N/A"

def invoke_conversational_model(prompt: str, model_id: str):
    """Send a conversational prompt via Bedrock converse API."""
    start_time = time.perf_counter()
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.5}
        )
        duration = time.perf_counter() - start_time
        usage = response.get('usage', {})
        answer_text = ""
        if 'output' in response and 'message' in response['output']:
            content = response['output']['message'].get('content', [])
            if content and 'text' in content[0]:
                answer_text = content[0]['text'].strip()
        return {
            "success": True,
            "latency_sec": duration,
            "input_tokens": usage.get('inputTokens', 0),
            "output_tokens": usage.get('outputTokens', 0),
            "answer_text": answer_text,
            "error": None
        }
    except ClientError as e:
        return {
            "success": False,
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "answer_text": "",
            "error": f"{e.response['Error']['Code']}: {e.response['Error']['Message']}"
        }

def get_embedding(text: str, model_id: str):
    """
    Get embedding vector for a single text.
    Returns (embedding_vector, latency_sec, input_tokens, success)
    """
    start_time = time.perf_counter()
    try:
        if "cohere" in model_id:
            body = json.dumps({"texts": [text], "input_type": "search_query"})
        else:  # Titan
            body = json.dumps({"inputText": text})
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        duration = time.perf_counter() - start_time
        response_body = json.loads(response['body'].read())
        
        # Extract embedding vector
        if "cohere" in model_id:
            embedding = response_body.get('embeddings', [])[0] if response_body.get('embeddings') else None
        else:  # Titan
            embedding = response_body.get('embedding')
        
        # Extract input token count
        if "amazon.titan" in model_id:
            input_tokens = response_body.get('inputTextTokenCount', 0)
        elif "cohere" in model_id:
            input_tokens = response_body.get('meta', {}).get('billed_units', {}).get('input_tokens', 0)
        else:
            input_tokens = 0
        
        return {
            "success": True,
            "embedding": embedding,
            "latency_sec": duration,
            "input_tokens": input_tokens,
            "error": None
        }
    except ClientError as e:
        return {
            "success": False,
            "embedding": None,
            "latency_sec": 0,
            "input_tokens": 0,
            "error": f"{e.response['Error']['Code']}: {e.response['Error']['Message']}"
        }

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

def is_correct(model_answer: str, keywords: list) -> bool:
    """Check if answer contains all keywords (case‑insensitive, punctuation ignored)."""
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

# ------------------------------------------------------------------------------
# 6. Test Execution for Conversational Models
# ------------------------------------------------------------------------------
def run_conversational_test_suite(model_id, delay=1.0, max_retries=3):
    """Run 20 conversational test cases, return results, correct count."""
    results = []
    correct_count = 0
    print(f"\n[Conversational] Testing Model: {model_id}")
    print(f"   Parameter size: {get_model_parameter_size(model_id)}")
    print(f"   Total Queries: {len(CONVERSATION_TEST_CASES)}")

    for i, test in enumerate(CONVERSATION_TEST_CASES):
        query = test["query"]
        keywords = test["keywords"]
        print(f"   Query {i+1}/{len(CONVERSATION_TEST_CASES)}: {query[:50]}...", end="\r")

        attempt = 0
        while attempt < max_retries:
            res = invoke_conversational_model(query, model_id)
            if res['success']:
                correct = is_correct(res['answer_text'], keywords)
                if correct:
                    correct_count += 1
                results.append({
                    **res,
                    "query": query,
                    "expected_keywords": keywords,
                    "correct": correct
                })
                break
            elif "Throttling" in str(res.get('error', '')):
                wait = (2 ** attempt) * 2
                print(f"\n   Throttled, waiting {wait}s...")
                time.sleep(wait)
                attempt += 1
            else:
                print(f"\n   Error: {res['error']}")
                results.append({**res, "query": query, "expected_keywords": keywords, "correct": False})
                break
        else:
            results.append({
                "success": False,
                "query": query,
                "expected_keywords": keywords,
                "correct": False,
                "error": "Max retries exceeded"
            })
        if i < len(CONVERSATION_TEST_CASES) - 1:
            time.sleep(delay)
    print(f"\n   Completed. Correct: {correct_count}/{len(CONVERSATION_TEST_CASES)}")
    return results, correct_count, len(CONVERSATION_TEST_CASES)

# ------------------------------------------------------------------------------
# 7. Test Execution for Embedding Models (with similarity evaluation)
# ------------------------------------------------------------------------------
def run_embedding_test_suite(model_id, test_pairs, delay=1.0, max_retries=3):
    """
    Run embedding test pairs: generate embeddings for both sentences,
    compute cosine similarity, compare to expected similarity.
    Returns: (list of detailed results, accuracy, total_latency_stats, token_stats)
    """
    results = []
    correct_pairs = 0
    latencies = []
    input_tokens_list = []
    
    print(f"\n[Embedding] Testing Model: {model_id}")
    print(f"   Total Pairs: {len(test_pairs)}")
    
    for i, (sent_a, sent_b, expected_similar) in enumerate(test_pairs):
        print(f"   Pair {i+1}/{len(test_pairs)}: {sent_a[:40]}... vs {sent_b[:40]}...", end="\r")
        
        # Get embedding for sentence A
        attempt_a = 0
        res_a = None
        while attempt_a < max_retries:
            res_a = get_embedding(sent_a, model_id)
            if res_a['success']:
                break
            elif "Throttling" in str(res_a.get('error', '')):
                wait = (2 ** attempt_a) * 2
                print(f"\n   Throttled, waiting {wait}s...")
                time.sleep(wait)
                attempt_a += 1
            else:
                break
        
        # Get embedding for sentence B
        attempt_b = 0
        res_b = None
        while attempt_b < max_retries:
            res_b = get_embedding(sent_b, model_id)
            if res_b['success']:
                break
            elif "Throttling" in str(res_b.get('error', '')):
                wait = (2 ** attempt_b) * 2
                print(f"\n   Throttled, waiting {wait}s...")
                time.sleep(wait)
                attempt_b += 1
            else:
                break
        
        if res_a and res_a['success'] and res_b and res_b['success']:
            # Compute similarity
            sim = cosine_similarity(res_a['embedding'], res_b['embedding'])
            # Determine if model considers them similar (threshold 0.5)
            predicted_similar = sim > 0.5
            is_correct = (predicted_similar == expected_similar)
            if is_correct:
                correct_pairs += 1
            latencies.append(res_a['latency_sec'])
            latencies.append(res_b['latency_sec'])
            input_tokens_list.append(res_a['input_tokens'])
            input_tokens_list.append(res_b['input_tokens'])
            results.append({
                "success": True,
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "expected_similar": expected_similar,
                "cosine_similarity": sim,
                "predicted_similar": predicted_similar,
                "correct": is_correct,
                "latency_a": res_a['latency_sec'],
                "latency_b": res_b['latency_sec'],
                "tokens_a": res_a['input_tokens'],
                "tokens_b": res_b['input_tokens']
            })
        else:
            results.append({
                "success": False,
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "expected_similar": expected_similar,
                "error_a": res_a.get('error') if res_a else "No response",
                "error_b": res_b.get('error') if res_b else "No response"
            })
        
        if i < len(test_pairs) - 1:
            time.sleep(delay)
    
    accuracy = (correct_pairs / len(test_pairs)) * 100 if test_pairs else 0
    print(f"\n   Completed. Similarity Accuracy: {correct_pairs}/{len(test_pairs)} ({accuracy:.1f}%)")
    return results, accuracy, latencies, input_tokens_list

# ------------------------------------------------------------------------------
# 8. Aggregate Statistics and Output
# ------------------------------------------------------------------------------
def print_conversational_stats(results, correct_count, total_queries, model_id):
    """Print per‑model summary for conversational models."""
    successful = [r for r in results if r.get('success', False)]
    correct_rate = (correct_count / total_queries) * 100 if total_queries > 0 else 0
    print("\n" + "="*50)
    print("CONVERSATIONAL MODEL RESULTS")
    print("="*50)
    print(f"Model ID:            {model_id}")
    print(f"Parameter Size:      {get_model_parameter_size(model_id)}")
    print(f"Successful Requests: {len(successful)}/{total_queries}")
    print(f"Correct Answers:     {correct_count}/{total_queries} ({correct_rate:.1f}%)")
    if successful:
        latencies = [r['latency_sec'] for r in successful]
        total_in = sum(r['input_tokens'] for r in successful)
        total_out = sum(r['output_tokens'] for r in successful)
        print(f"Average Latency:     {mean(latencies):.4f} sec")
        print(f"Min Latency:         {min(latencies):.4f} sec")
        print(f"Max Latency:         {max(latencies):.4f} sec")
        if len(latencies) > 1:
            print(f"Std Deviation:       {stdev(latencies):.4f} sec")
        print(f"Total Input Tokens:  {total_in}")
        print(f"Total Output Tokens: {total_out}")
    else:
        print("No successful requests.")
    print("="*50)
    return {
        "type": "conversational",
        "model_id": model_id,
        "param_size": get_model_parameter_size(model_id),
        "successful": len(successful),
        "total": total_queries,
        "correct": correct_count,
        "correct_rate": correct_rate,
        "avg_latency": mean([r['latency_sec'] for r in successful]) if successful else None,
        "min_latency": min([r['latency_sec'] for r in successful]) if successful else None,
        "max_latency": max([r['latency_sec'] for r in successful]) if successful else None,
        "total_input_tokens": sum(r['input_tokens'] for r in successful) if successful else 0,
        "total_output_tokens": sum(r['output_tokens'] for r in successful) if successful else 0,
    }

def print_embedding_stats(results, accuracy, latencies, input_tokens_list, model_id):
    """Print per‑model summary for embedding models (including similarity accuracy)."""
    successful = [r for r in results if r.get('success', False)]
    print("\n" + "="*50)
    print("EMBEDDING MODEL RESULTS")
    print("="*50)
    print(f"Model ID:                 {model_id}")
    print(f"Successful Pairs:         {len(successful)}/{len(results)}")
    print(f"Semantic Similarity Acc:  {accuracy:.1f}% ({len([r for r in successful if r.get('correct')])}/{len(successful)} correct)")
    if latencies:
        print(f"Average Latency (per text): {mean(latencies):.4f} sec")
        print(f"Min Latency:                {min(latencies):.4f} sec")
        print(f"Max Latency:                {max(latencies):.4f} sec")
        if len(latencies) > 1:
            print(f"Std Deviation:              {stdev(latencies):.4f} sec")
        print(f"Total Input Tokens:         {sum(input_tokens_list)}")
    else:
        print("No successful requests.")
    print("="*50)
    return {
        "type": "embedding",
        "model_id": model_id,
        "successful": len(successful),
        "total": len(results),
        "similarity_accuracy": accuracy,
        "correct_pairs": len([r for r in successful if r.get('correct')]),
        "avg_latency": mean(latencies) if latencies else None,
        "min_latency": min(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
        "total_input_tokens": sum(input_tokens_list) if input_tokens_list else 0,
    }

def print_combined_summary(conv_summaries, embed_summaries):
    """Print a combined table of all models (conversational + embedding)."""
    print("\n" + "="*110)
    print("FINAL SUMMARY: LATENCY & TOKEN USAGE (ap-southeast-2)")
    print("="*110)
    # Conversational header
    print(f"{'Type':<12} {'Model ID':<40} {'Succ Rate':<10} {'Correct Rate':<12} {'Avg Lat (s)':<12} {'In Tokens':<10} {'Out Tokens':<10}")
    print("-"*110)
    for s in conv_summaries:
        print(f"{'Conv':<12} {s['model_id']:<40} {s['successful']}/{s['total']:<9} {s['correct']}/{s['total']} ({s['correct_rate']:.1f}%)   {s['avg_latency']:<12.4f} {s['total_input_tokens']:<10} {s['total_output_tokens']:<10}")
    # Embedding header
    print("-"*110)
    print(f"{'Type':<12} {'Model ID':<40} {'Succ Rate':<10} {'Similarity Acc':<14} {'Avg Lat (s)':<12} {'In Tokens':<10}")
    print("-"*110)
    for e in embed_summaries:
        print(f"{'Embed':<12} {e['model_id']:<40} {e['successful']}/{e['total']:<9} {e['similarity_accuracy']:.1f}% ({e['correct_pairs']}/{e['total']})   {e['avg_latency']:<12.4f} {e['total_input_tokens']:<10}")
    print("="*110)

def write_results_to_csv(all_summaries, folder_name="experiment_results"):
    """Write all model summaries (conversational + embedding) to CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(parent_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bedrock_latency_evaluation_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model Type", "Model ID", "Parameter Size", "Success Rate",
                         "Avg Latency (s)", "Min Latency (s)", "Max Latency (s)",
                         "Total Input Tokens", "Total Output Tokens",
                         "Correct Answers / Total", "Correct Rate (%)",
                         "Similarity Accuracy (%)", "Correct Pairs / Total"])
        for s in all_summaries:
            if s["type"] == "conversational":
                success_rate = f"{s['successful']}/{s['total']}"
                correct_str = f"{s['correct']}/{s['total']}"
                correct_rate = f"{s['correct_rate']:.1f}"
                output_tokens = s['total_output_tokens']
                sim_acc = "N/A"
                correct_pairs = "N/A"
            else:  # embedding
                success_rate = f"{s['successful']}/{s['total']}"
                correct_str = "N/A"
                correct_rate = "N/A"
                output_tokens = 0
                sim_acc = f"{s['similarity_accuracy']:.1f}"
                correct_pairs = f"{s['correct_pairs']}/{s['total']}"
            writer.writerow([
                s["type"],
                s["model_id"],
                s.get("param_size", "N/A"),
                success_rate,
                f"{s['avg_latency']:.4f}" if s['avg_latency'] is not None else "N/A",
                f"{s['min_latency']:.4f}" if s['min_latency'] is not None else "N/A",
                f"{s['max_latency']:.4f}" if s['max_latency'] is not None else "N/A",
                s['total_input_tokens'],
                output_tokens,
                correct_str,
                correct_rate,
                sim_acc,
                correct_pairs
            ])
    print(f"\nResults saved to: {filepath}")

# ------------------------------------------------------------------------------
# 9. Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AWS BEDROCK LATENCY, TOKEN & SIMILARITY EVALUATION")
    print("(Conversational + Embedding Models)")
    print("="*70)
    
    conv_summaries = []
    embed_summaries = []

    # 1. Test conversational models
    print("\n" + "="*70)
    print("PHASE 1: CONVERSATIONAL MODELS")
    print("="*70)
    for model_id in CONVERSATION_MODEL_IDS:
        results, correct_count, total_q = run_conversational_test_suite(model_id, REQUEST_DELAY, MAX_RETRIES)
        summary = print_conversational_stats(results, correct_count, total_q, model_id)
        conv_summaries.append(summary)

    # 2. Test embedding models (with similarity evaluation)
    print("\n" + "="*70)
    print("PHASE 2: EMBEDDING MODELS (Semantic Similarity)")
    print("="*70)
    for model_id in EMBEDDING_MODEL_IDS:
        results, accuracy, latencies, tokens_list = run_embedding_test_suite(model_id, EMBEDDING_TEST_PAIRS, REQUEST_DELAY, MAX_RETRIES)
        summary = print_embedding_stats(results, accuracy, latencies, tokens_list, model_id)
        embed_summaries.append(summary)
    
    # 3. Print combined summary table
    print_combined_summary(conv_summaries, embed_summaries)
    
    # 4. Save all results to CSV
    all_summaries = conv_summaries + embed_summaries
    write_results_to_csv(all_summaries, "experiment_results")

    print("\nEvaluation complete.")