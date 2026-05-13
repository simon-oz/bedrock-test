import boto3
import time
import os 
import re
from botocore.exceptions import ClientError
from statistics import mean, stdev
from datetime import datetime

# 1. Setup Clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="ap-southeast-2")
bedrock_management = boto3.client(service_name="bedrock", region_name="ap-southeast-2")  # for listing models

# 2. Configuration
MODEL_IDs = [
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

    # "mistral.voxtral-mini-3b-2507",
    # "twelvelabs.pegasus-1-2-v1:0",
    
    # coding
    # " mistral.devstral-2-123b", 
]

REQUEST_DELAY = 1.0
MAX_RETRIES = 3


# ------------------------------
# 3. Expanded Test Data (20 queries with reference keywords and source URLs)
# ------------------------------
TEST_CASES = [
    {
        "query": "What is the capital of Australia?",
        "keywords": ["canberra"],
        "source": "https://www.australia.com/en/facts-and-planning/about-australia/cities/canberra.html"
    },
    {
        "query": "Who painted the Mona Lisa?",
        "keywords": ["leonardo", "da vinci"],
        "source": "https://www.louvre.fr/en/explore/the-palace/from-the-chateau-to-the-museum-mona-lisa"
    },
    {
        "query": "If a train travels at 60 km/h for 2 hours, how far does it go?",
        "keywords": ["120", "km"],
        "source": "https://www.mathsisfun.com/measure/speed-distance-time.html"
    },
    {
        "query": "Solve for x: 2x + 5 = 13",
        "keywords": ["x = 4", "x=4"],
        "source": "https://www.mathsisfun.com/algebra/equations-solving.html"
    },
    {
        "query": "What is compound interest?",
        "keywords": ["interest on interest", "interest on principal"],
        "source": "https://www.investopedia.com/terms/c/compoundinterest.asp"
    },
    {
        "query": "What does 'habeas corpus' mean?",
        "keywords": ["produce the body", "bring the body"],
        "source": "https://www.law.cornell.edu/wex/habeas_corpus"
    },
    {
        "query": "Explain the theory of relativity in simple terms.",
        "keywords": ["einstein", "space", "time", "gravity"],
        "source": "https://www.space.com/17661-theory-general-relativity.html"
    },
    {
        "query": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "keywords": ["no", "cannot", "not necessarily"],
        "source": "https://www.logicmountain.com/fallacies/quantifier-fallacy"
    },
    {
        "query": "When did World War II end?",
        "keywords": ["1945"],
        "source": "https://www.history.com/topics/world-war-ii/world-war-ii-history"
    },
    {
        "query": "What is photosynthesis?",
        "keywords": ["plants", "sunlight", "energy", "carbon dioxide", "oxygen"],
        "source": "https://www.nationalgeographic.org/encyclopedia/photosynthesis/"
    },
    {
        "query": "What is inflation?",
        "keywords": ["general increase", "prices", "purchasing power"],
        "source": "https://www.imf.org/en/Publications/fandd/issues/Series/Back-to-Basics/Inflation"
    },
    {
        "query": "What is a transformer in machine learning?",
        "keywords": ["attention", "sequence", "neural network"],
        "source": "https://arxiv.org/abs/1706.03762"
    },
    {
        "query": "What is the function of the human liver?",
        "keywords": ["detoxification", "protein synthesis", "bile"],
        "source": "https://www.britannica.com/science/liver"
    },
    {
        "query": "Which river is the longest in the world?",
        "keywords": ["nile"],
        "source": "https://www.britannica.com/story/is-the-nile-the-longest-river-in-the-world"
    },
    {
        "query": "Who was the first President of the United States?",
        "keywords": ["george washington"],
        "source": "https://www.whitehouse.gov/about-the-white-house/presidents/george-washington/"
    },
    {
        "query": "Who wrote 'Pride and Prejudice'?",
        "keywords": ["jane austen"],
        "source": "https://www.britannica.com/biography/Jane-Austen"
    },
    {
        "query": "What is the 'categorical imperative'?",
        "keywords": ["kant", "universal law", "duty", "moral"],
        "source": "https://plato.stanford.edu/entries/kant-moral/"
    },
    {
        "query": "Who won the FIFA World Cup in 2022?",
        "keywords": ["argentina"],
        "source": "https://www.fifa.com/tournaments/mens/worldcup/qatar2022"
    },
    {
        "query": "What is 'due process'?",
        "keywords": ["fair treatment", "judicial", "legal proceedings"],
        "source": "https://www.law.cornell.edu/wex/due_process"
    },
    {
        "query": "What is the probability of rolling a 6 on a fair die?",
        "keywords": ["1/6", "one sixth", "0.1667"],
        "source": "https://www.mathsisfun.com/data/probability.html"
    }
]

# ------------------------------
# 4. Helper Functions
# ------------------------------
def get_model_parameter_size(model_id: str) -> str:
    """Retrieve model parameter size from Bedrock API."""
    try:
        response = bedrock_management.list_foundation_models()
        model_info = None
        for model in response.get('modelSummaries', []):
            if model.get('modelId') == model_id:
                model_info = model
                break
        if not model_info:
            return "Unknown (Not Found)"
        model_name = model_info.get('modelName', '').lower()
        name_to_check = f"{model_name} {model_id}".lower()
        match = re.search(r'(\d+(?:\.\d+)?)b', name_to_check)
        if match:
            return f"{match.group(1)}B"
        if "glm-4" in model_id:
            return "4.7B"
        if "qwen3" in model_id and "32b" in model_id:
            return "32B"
        if "gpt-oss-20b" in model_id:
            return "20B"
        return "Unknown"
    except Exception as e:
        return f"Unknown ({str(e)[:20]})"

def invoke_bedrock(prompt, model_id):
    """Send request to Bedrock Runtime, return latency, token info, and response text."""
    start_time = time.perf_counter()
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.5}
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        usage = response.get('usage', {})
        # Extract the model's answer text
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
            "error": None,
            "answer_text": answer_text
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        err_msg = e.response['Error']['Message']
        return {
            "success": False,
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": f"{error_code}: {err_msg}",
            "answer_text": ""
        }

def is_correct(model_answer: str, keywords: list) -> bool:
    """
    Check if the model's answer contains all required keywords (case-insensitive,
    ignoring punctuation and common filler words).
    """
    if not model_answer:
        return False
    # Normalize: lower case, remove punctuation, split into words
    cleaned = re.sub(r'[^\w\s]', '', model_answer.lower())
    words = set(cleaned.split())
    for kw in keywords:
        kw_lower = kw.lower()
        # If keyword contains spaces, check as phrase
        if ' ' in kw_lower:
            if kw_lower not in cleaned:
                return False
        else:
            if kw_lower not in words:
                return False
    return True

# ------------------------------
# 5. Test Execution for One Model
# ------------------------------
def run_test_suite(model_id, delay):
    """Run all 20 test cases on a single model, return results and correctness count."""
    results = []
    correct_count = 0
    param_size = get_model_parameter_size(model_id)
    print(f"\nStarting test for Model: {model_id}")
    print(f"Parameter size: {param_size}")
    print(f"Total Queries: {len(TEST_CASES)}")

    for i, test in enumerate(TEST_CASES):
        query = test["query"]
        keywords = test["keywords"]
        print(f"Processing Query {i+1}/{len(TEST_CASES)}: {query[:50]}...", end="\r")

        attempt = 0
        while attempt < MAX_RETRIES:
            res = invoke_bedrock(query, model_id)
            if res['success']:
                # Evaluate correctness
                correct = is_correct(res['answer_text'], keywords)
                if correct:
                    correct_count += 1
                # Append detailed result for potential post-analysis
                results.append({
                    **res,
                    "query": query,
                    "expected_keywords": keywords,
                    "correct": correct
                })
                break
            elif "Throttling" in res['error']:
                wait_time = (2 ** attempt) * 2
                print(f"\nThrottled on query {i+1}. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                attempt += 1
            else:
                print(f"\nError on query {i+1}: {res['error']}")
                results.append({**res, "query": query, "expected_keywords": keywords, "correct": False})
                break
        else:
            # Max retries exhausted
            results.append({
                "success": False,
                "query": query,
                "expected_keywords": keywords,
                "correct": False,
                "error": "Max retries exceeded"
            })
        if i < len(TEST_CASES) - 1:
            time.sleep(delay)
    print("\nTest completed.")
    return results, correct_count, param_size

def print_stats(results, correct_count, total_queries, model_id, param_size):
    """Print per‑model summary and return aggregated data for final table."""
    successful = [r for r in results if r.get('success', False)]
    correct_rate = (correct_count / total_queries) * 100 if total_queries > 0 else 0
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Model ID:            {model_id}")
    print(f"Parameter Size:      {param_size}")
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
        print("No successful requests to compute latency/tokens.")
    print("="*50)
    return {
        "model_id": model_id,
        "param_size": param_size,
        "successful": len(successful),
        "total": total_queries,
        "correct": correct_count,
        "correct_rate": correct_rate,
        "avg_latency": mean([r['latency_sec'] for r in successful]) if successful else None,
        "min_latency": min([r['latency_sec'] for r in successful]) if successful else None,
        "max_latency": max([r['latency_sec'] for r in successful]) if successful else None,
        "total_input_tokens": sum(r['input_tokens'] for r in successful) if successful else 0,
        "total_output_tokens": sum(r['output_tokens'] for r in successful) if successful else 0
    }

def print_final_summary(all_summaries):
    """Print a final table with all models and their correct rates, plus latency stats."""
    print("\n\n" + "="*140)
    print("FINAL SUMMARY TABLE FOR ALL MODELS")
    print("="*140)
    header = (f"{'Model ID':<45} {'Params':<15} {'Succ Rate':<10} {'Correct Rate':<12} "
              f"{'Avg Lat (s)':<12} {'Min Lat (s)':<12} {'Max Lat (s)':<12} "
              f"{'In Tokens':<12} {'Out Tokens':<12}")
    print(header)
    print("-" * 140)
    for s in all_summaries:
        success_rate = f"{s['successful']}/{s['total']}"
        correct_rate = f"{s['correct']}/{s['total']} ({s['correct_rate']:.1f}%)"
        avg_lat = f"{s['avg_latency']:.4f}" if s['avg_latency'] is not None else "N/A"
        min_lat = f"{s['min_latency']:.4f}" if s['min_latency'] is not None else "N/A"
        max_lat = f"{s['max_latency']:.4f}" if s['max_latency'] is not None else "N/A"
        print(f"{s['model_id']:<45} {s['param_size']:<15} {success_rate:<10} {correct_rate:<12} "
              f"{avg_lat:<12} {min_lat:<12} {max_lat:<12} "
              f"{s['total_input_tokens']:<12} {s['total_output_tokens']:<12}")
    print("="*140)


def write_results_to_csv(all_summaries, folder_name="experiment_results"):
    """Write the final summary results to a CSV file in the specified folder (one level above the script)."""
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up
    parent_dir = os.path.dirname(script_dir)
    # Build full path for the results folder
    results_dir = os.path.join(parent_dir, folder_name)
    # Create folder if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bedrock_experiment_results_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Define CSV columns
    fieldnames = [
        "model_id",
        "param_size",
        "successful_requests",
        "total_requests",
        "correct_answers",
        "correct_rate_percent",
        "avg_latency_sec",
        "min_latency_sec",
        "max_latency_sec",
        "total_input_tokens",
        "total_output_tokens"
    ]
    
    with open(filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            # Prepare row (convert None to empty string for CSV)
            row = {
                "model_id": s["model_id"],
                "param_size": s["param_size"],
                "successful_requests": s["successful"],
                "total_requests": s["total"],
                "correct_answers": s["correct"],
                "correct_rate_percent": round(s["correct_rate"], 2) if s["correct_rate"] is not None else "",
                "avg_latency_sec": round(s["avg_latency"], 4) if s["avg_latency"] is not None else "",
                "min_latency_sec": round(s["min_latency"], 4) if s["min_latency"] is not None else "",
                "max_latency_sec": round(s["max_latency"], 4) if s["max_latency"] is not None else "",
                "total_input_tokens": s["total_input_tokens"],
                "total_output_tokens": s["total_output_tokens"]
            }
            writer.writerow(row)
    
    print(f"\nResults successfully written to: {filepath}")

# ------------------------------
# 6. Main Execution
# ------------------------------
if __name__ == "__main__":
    all_model_summaries = []
    for model_id in MODEL_IDs:
        results, correct_count, param_size = run_test_suite(model_id, REQUEST_DELAY)
        summary = print_stats(results, correct_count, len(TEST_CASES), model_id, param_size)
        all_model_summaries.append(summary)
    
    # Print final summary to console
    print_final_summary(all_model_summaries)
    
    # Write results to CSV in ./experiment_results (one level above the script)
    write_results_to_csv(all_model_summaries, "experiment_results")
