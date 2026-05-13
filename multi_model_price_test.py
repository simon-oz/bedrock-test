import boto3
import time
import json
import re
import os
import csv

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
# 2. Pricing per 1,000 tokens (ap-southeast-2) – all models filled
#    Sources: AWS Bedrock Pricing page (April 2026)
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
}
# Default fallback (used if model not found – will print warning)
DEFAULT_PRICING = {"input": 0.0010, "output": 0.0030}

# Embedding models (special handling)
EMBEDDING_MODELS = {
    "cohere.embed-english-v3",
    "amazon.titan-embed-text-v2:0",
    "cohere.embed-multilingual-v3",
}

# -----------------------------------------------------------------------------
# 3. Model List (full)
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
# 4. Test Data – 20 queries with reference keywords and sources
# -----------------------------------------------------------------------------
TEST_CASES = [
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

REQUEST_DELAY = 1.0      # seconds between requests per model
MODEL_DELAY = 5.0        # seconds between different models
MAX_RETRIES = 3

# -----------------------------------------------------------------------------
# 5. Helper Functions
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
            # minimal embed test
            if "cohere" in model_id:
                body = json.dumps({"texts": ["test"], "input_type": "search_query"})
            else:  # Titan
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

def is_correct(model_answer: str, keywords: List[str]) -> bool:
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

def invoke_bedrock(prompt: str, model_id: str) -> Dict[str, Any]:
    """Send a single prompt, return latency, tokens, cost, answer text."""
    start = time.perf_counter()
    try:
        if model_id in EMBEDDING_MODELS:
            # Embedding path – no output text, token count only
            if "cohere" in model_id:
                body = json.dumps({"texts": [prompt], "input_type": "search_query"})
            else:
                body = json.dumps({"inputText": prompt})
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            resp_body = json.loads(response['body'].read())
            if "amazon.titan" in model_id:
                in_tokens = resp_body.get('inputTextTokenCount', 0)
            elif "cohere" in model_id:
                in_tokens = resp_body.get('meta', {}).get('billed_units', {}).get('input_tokens', 0)
            else:
                in_tokens = 0
            out_tokens = 0
            answer_text = "(embedding output omitted)"
        else:
            # Conversational
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.5}
            )
            usage = response.get('usage', {})
            in_tokens = usage.get('inputTokens', 0)
            out_tokens = usage.get('outputTokens', 0)
            content = response.get('output', {}).get('message', {}).get('content', [])
            answer_text = content[0].get('text', '') if content else ''
        latency = time.perf_counter() - start
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

def run_test_suite(model_id: str) -> Tuple[List[Dict], int, int, float]:
    """
    Run all test cases against one model.
    Returns: (list_of_results, correct_count, total_queries, total_cost)
    """
    correct = 0
    results = []
    total_cost = 0.0
    print(f"\nTesting: {model_id} (size {get_model_parameter_size(model_id)})")
    pricing = get_model_pricing(model_id)
    print(f"Pricing: ${pricing['input']}/1K input, ${pricing['output']}/1K output")
    # Access check
    accessible, err = check_model_access(model_id)
    if not accessible:
        print(f"Model not accessible: {err}")
        return results, 0, len(TEST_CASES), 0.0
    print("Model accessible")
    for idx, test in enumerate(TEST_CASES, 1):
        print(f"Query {idx}/{len(TEST_CASES)}: {test['query'][:50]}...", end="\r")
        attempt = 0
        res = None
        while attempt < MAX_RETRIES:
            res = invoke_bedrock(test['query'], model_id)
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
            # Evaluate correctness
            if is_correct(res['answer_text'], test['keywords']):
                correct += 1
            else:
                # Optionally print mismatch for debugging (commented)
                # print(f"\n   ✗ Expected keywords: {test['keywords']}")
                pass
            results.append(res)
            total_cost += res['total_cost']
        else:
            # Failed request
            results.append({
                "success": False,
                "query": test['query'],
                "error": res.get('error', 'Unknown') if res else "No response",
                "total_cost": 0
            })
        time.sleep(REQUEST_DELAY)
    print(f"\nCompleted. Correct: {correct}/{len(TEST_CASES)}")
    return results, correct, len(TEST_CASES), total_cost

def aggregate_stats(results: List[Dict], correct_count: int, total_queries: int) -> Dict[str, Any]:
    """Compute aggregated stats including correctness, latency, tokens, cost."""
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        # No successful requests – return zeros but keep correctness info
        return {
            "success_rate": 0.0,
            "correct_rate": (correct_count / total_queries) * 100,
            "correct_count": correct_count,          # <-- ADDED
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
        "correct_count": correct_count,              # <-- ADDED (already present here)
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


def print_final_table(all_stats: Dict[str, Dict[str, Any]]):
    """Print a compact comparison table."""
    print("\n" + "="*130)
    print("FINAL COMPARISON TABLE (ap-southeast-2)")
    print("="*130)
    header = (f"{'Model':<45} {'Size':<6} {'Correct':<9} {'Latency(avg)':<12} "
              f"{'TPS':<8} {'Cost/Req':<12} {'Cost/1K':<12} {'Success':<8}")
    print(header)
    print("-"*130)
    for model_id, stats in sorted(all_stats.items(), key=lambda x: x[1]['correct_rate'], reverse=True):
        short_name = model_id if len(model_id) <= 44 else model_id[:41] + "..."
        correct_str = f"{stats['correct_count']}/20 ({stats['correct_rate']:.0f}%)"
        succ_str = f"{stats['success_rate']:.0f}%"
        print(f"{short_name:<45} {get_model_parameter_size(model_id):<6} {correct_str:<9} "
              f"{stats['avg_latency']:<12.3f} {stats['tokens_per_second']:<8.1f} "
              f"${stats['avg_cost_per_req']:<11.6f} ${stats['cost_per_1k_tokens']:<11.6f} {succ_str:<8}")
    print("="*130)

def save_full_results(all_raw: Dict[str, List[Dict]], all_stats: Dict[str, Dict[str, Any]]):
    """Save per‑request details and aggregated summary to CSV in ../experiment_results/"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(parent_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bedrock_full_evaluation_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "query_id", "success", "latency_sec", "input_tokens",
                         "output_tokens", "input_cost", "output_cost", "total_cost", "error"])
        for model_id, results in all_raw.items():
            for i, r in enumerate(results, 1):
                writer.writerow([
                    model_id, i, r.get('success', False),
                    round(r.get('latency_sec', 0), 4),
                    r.get('input_tokens', 0), r.get('output_tokens', 0),
                    round(r.get('input_cost', 0), 8), round(r.get('output_cost', 0), 8),
                    round(r.get('total_cost', 0), 8), r.get('error', '')
                ])
        # Append summary rows
        writer.writerow([])
        writer.writerow(["SUMMARY_PER_MODEL"])
        writer.writerow(["model_id", "param_size", "success_rate%", "correct_rate%",
                         "avg_latency", "min_latency", "max_latency", "total_cost_usd",
                         "avg_cost_per_req", "cost_per_1k_tokens", "tokens_per_second"])
        for model_id, s in all_stats.items():
            writer.writerow([
                model_id, get_model_parameter_size(model_id), f"{s['success_rate']:.1f}",
                f"{s['correct_rate']:.1f}", f"{s['avg_latency']:.4f}", f"{s['min_latency']:.4f}",
                f"{s['max_latency']:.4f}", f"{s['total_cost']:.6f}", f"{s['avg_cost_per_req']:.8f}",
                f"{s['cost_per_1k_tokens']:.6f}", f"{s['tokens_per_second']:.2f}"
            ])
    print(f"\nFull results (per‑request + summary) saved to: {filepath}")

def main():
    print("\nAWS Bedrock Model Evaluation (Latency, Cost, Correctness)")
    print(f"   Region: ap-southeast-2 | Models: {len(MODEL_IDS)} | Queries: {len(TEST_CASES)}")
    all_raw = {}
    all_stats = {}
    total_test_cost = 0.0
    for idx, model_id in enumerate(MODEL_IDS, 1):
        results, correct, total_q, model_cost = run_test_suite(model_id)
        all_raw[model_id] = results
        stats = aggregate_stats(results, correct, total_q)
        all_stats[model_id] = stats
        total_test_cost += model_cost
        if idx < len(MODEL_IDS):
            print(f"\n Waiting {MODEL_DELAY}s before next model...")
            time.sleep(MODEL_DELAY)
    print_final_table(all_stats)
    print(f"\n Total estimated test cost: ${total_test_cost:.6f}")
    print("Note: Costs are estimates based on AWS Bedrock pricing (ap-southeast-2).")
    save_full_results(all_raw, all_stats)
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()