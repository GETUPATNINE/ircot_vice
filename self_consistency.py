import os
import json
import re
import time
import requests
import socket
import http.client
from requests.exceptions import RequestException
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter # Import Counter for voting
from ircot_retrive import IRCoTSystem
from typing import Tuple

# --- 1. Centralized Configuration ---
load_dotenv()
os.environ["GLOBALAI_API_KEY"] = "" # gpt-3.5-turbo api key
ZHIPU_API_KEY = "" # glm-z1-flash api key

# API Configuration
API_KEY = os.getenv("GLOBALAI_API_KEY")
API_HOST = "globalai.vip"
API_PATH = "/v1/chat/completions"
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7 # Temperature > 0 is crucial for self-consistency
API_TIMEOUT = 90 # seconds

# Retry Configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5 # seconds

# Evaluation Configuration
DATASET_NAME = "deepmind/aqua_rat"
DATASET_SPLIT = "test"
OUTPUT_FILE = "evaluation_results_self_consistency.jsonl"
SELF_CONSISTENCY_SAMPLES = 5 # <<-- NEW: Number of samples for self-consistency

# --- 2. Prompt Engineering ---
PROMPT_TEMPLATE = """
You are an expert mathematician and a meticulous problem solver. Your task is to solve the following algebraic word problem.

First, think step-by-step to arrive at the solution. Lay out your reasoning clearly.
After you have determined the answer, and only then, state your final choice on a new line.

### Question:
{question}

### Options:
{options}

Provide your final answer on a new line in the format: `Final Answer: [X]` where X is one of A, B, C, D, or E.
"""

PROMPT_TEMPLATE_WITH_IRCoT = """
You are an expert mathematician and a meticulous problem solver. Your task is to solve the following algebraic word problem.

First, think step-by-step to arrive at the solution. Lay out your reasoning clearly. There are some retrieved contents that may help you solve the problem, you can use them to help you solve the problem. But you should not use them as your final answer.
After you have determined the answer, and only then, state your final choice on a new line.

### Question:
{question}

### Options:
{options}

### CoT Reasoning:
{cot_reasoning}

### Retrieved Contents:
{retrieved_contents}

Provide your final answer on a new line in the format: `Final Answer: [X]` where X is one of A, B, C, D, or E.
"""

def format_prompt(problem: dict, ircot_system: IRCoTSystem) -> str:
    """Formats a problem from the dataset into a structured prompt."""
    question_text = problem['question']
    options_text = "\n".join(problem['options'])
    result = ircot_system.run(problem['question'], problem['options'])
    return PROMPT_TEMPLATE_WITH_IRCoT.format(question=result['question'], options=result['options'], cot_reasoning=result['cot_reasoning'], retrieved_contents=result['retrieved_contents'])

# --- 3. Robust API Client (Unchanged) ---
# Your original call_llm_api function is used here without modification.
# It's robust and will be called multiple times by the new self-consistency function.
def call_llm_api(prompt: str) -> Tuple[str, str]:
    """
    使用具有弹性重试机制的 http.client 调用语言模型 API。
    返回一个元组 (content, error_message)。
    """
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "stream": False,
    })

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'Host': API_HOST,
        'Connection': 'keep-alive'
    }

    for attempt in range(MAX_RETRIES):
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=API_TIMEOUT)
            conn.request("POST", API_PATH, payload, headers)
            res = conn.getresponse()
            
            if res.status != 200:
                error_body = res.read().decode('utf-8')
                raise Exception(f"API request failed, status code: {res.status}, response: {error_body}")

            data = res.read()
            conn.close()
            
            response_json = json.loads(data.decode("utf-8"))
            
            choice = response_json.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            
            if not content:
                return None, "API_EMPTY_CONTENT"
            
            return content.strip(), None

        except (http.client.HTTPException, socket.gaierror, socket.timeout, ConnectionRefusedError, Exception) as e:
            error_msg = f"API_REQUEST_FAILED: {e}"
            print(f"警告：第 {attempt + 1}/{MAX_RETRIES} 次 API 调用失败。错误: {error_msg}")
            
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"{delay} 秒后重试...")
                time.sleep(delay)
            else:
                return None, error_msg
                
    return None, "API_MAX_RETRIES_EXCEEDED"

# --- 4. Output Parsing and Normalization (Unchanged) ---
def parse_model_output(output: str) -> str:
    """
    Parses the model's raw text output to extract the final answer.
    Returns the choice ('A', 'B', 'C', 'D', 'E') or a parse error code.
    """
    if not output:
        return "PARSE_NO_OUTPUT"

    match = re.search(r"Final Answer:\s*\[?([A-E])\]?", output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    found_letters = re.findall(r'\b([A-E])\b', output)
    if found_letters:
        return found_letters[-1].upper()

    return "PARSE_FAILURE"

# --- 5. NEW: Self-Consistency Logic ---
def get_self_consistent_answer(prompt: str) -> Tuple[str, list, list, list]:
    """
    Calls the LLM API multiple times and returns the majority vote answer.

    Args:
        prompt: The input prompt for the model.

    Returns:
        A tuple containing:
        - final_answer (str): The most common parsed answer, or a VOTE_FAILED code.
        - all_raw_outputs (list): A list of all raw text outputs from the API.
        - all_parsed_answers (list): A list of all parsed answers.
        - all_errors (list): A list of any errors encountered during API calls.
    """
    all_raw_outputs = []
    all_parsed_answers = []
    all_errors = []

    for _ in range(SELF_CONSISTENCY_SAMPLES):
        raw_output, error_message = call_llm_api(prompt)
        all_raw_outputs.append(raw_output)
        all_errors.append(error_message)

        if raw_output:
            parsed_answer = parse_model_output(raw_output)
            all_parsed_answers.append(parsed_answer)
        else:
            # Append the error code if the output was empty
            all_parsed_answers.append(error_message or "API_ERROR")

    # Filter out parsing errors or API errors before voting
    valid_answers = [ans for ans in all_parsed_answers if not ans.startswith("PARSE_") and not ans.startswith("API_")]

    if not valid_answers:
        return "VOTE_FAILED", all_raw_outputs, all_parsed_answers, all_errors

    # Use collections.Counter to find the most common answer
    vote_counts = Counter(valid_answers)
    final_answer = vote_counts.most_common(1)[0][0]

    return final_answer, all_raw_outputs, all_parsed_answers, all_errors

# --- 6. Main Evaluation Orchestrator (MODIFIED) ---
def main():
    """Main function to run the evaluation with self-consistency."""

    if not API_KEY:
        print("Error: GLOBALAI_API_KEY environment variable not set. Exiting.")
        return

    print(f"Loading dataset '{DATASET_NAME}' split '{DATASET_SPLIT}'...")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    results = []
    correct_count = 0
    total_count = 0

    print("Loading IRCoT system...")
    ircot_system = IRCoTSystem(
        corpus_name="aqua_train",
        max_iterations=3,
        corpus_path="AQuA/train.tok.json",
        initial_retrieval_count=5,  # 初次检索的相似问题数量
        iterative_retrieval_count=3,  # 迭代检索的相似问题数量
        api_key=ZHIPU_API_KEY,
        model_name="glm-z1-flash"
    )

    print(f"Starting evaluation on {len(dataset)} problems with {SELF_CONSISTENCY_SAMPLES} samples each...")
    # Open file in append mode to save progress incrementally
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i, problem in enumerate(tqdm(dataset, desc="Evaluating Problems")):
            total_count += 1
            
            # 1. Prompt
            prompt = format_prompt(problem, ircot_system)
            
            # 2. Query (with Self-Consistency)
            final_voted_answer, all_raw_outputs, all_parsed_answers, all_errors = get_self_consistent_answer(prompt)

            # 3. Compare
            ground_truth = problem['correct'].strip().upper()
            is_correct = (final_voted_answer == ground_truth)
            if is_correct:
                correct_count += 1

            # 4. Log (with detailed self-consistency results)
            result_log = {
                "problem_id": i,
                "question": problem['question'],
                "options": problem['options'],
                "ground_truth": ground_truth,
                "final_voted_answer": final_voted_answer,
                "is_correct": is_correct,
                "self_consistency_results": {
                    "parsed_answers": all_parsed_answers,
                    "raw_outputs": all_raw_outputs,
                    "errors": [err for err in all_errors if err is not None]
                }
            }
            f.write(json.dumps(result_log) + "\n")
            results.append(result_log)

    # --- 7. Final Aggregation and Reporting ---
    print("\n--- Evaluation Complete ---")
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"Total Problems Evaluated: {total_count}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")

        # Deeper analysis of failures
        vote_failures = sum(1 for r in results if r['final_voted_answer'] == "VOTE_FAILED")
        
        print(f"\nBreakdown of Failures:")
        print(f"  Vote Failures (no valid answer from any sample): {vote_failures} ({ (vote_failures/total_count)*100:.2f}%)")
        print(f"Note: See the JSONL file for detailed API and parsing errors per sample.")
    else:
        print("No problems were evaluated.")

    print(f"Detailed results have been saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()