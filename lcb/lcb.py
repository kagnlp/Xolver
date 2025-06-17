import openai
import json
import random
import re
import subprocess
import tempfile
import os
import pickle
import json
import sys
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset

nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    print("Downloading 'punkt_tab' resource for NLTK...")
    nltk.download('punkt_tab') 

test = load_dataset("livecodebench/code_generation")
episodic_memory = load_dataset("krarit/retrivcod")

# ========== PROMPT TEMPLATES ==========
PLANNER_PROMPT = """
You are a planner to solve a {task_type} problem. Here is the problem for which you have to plan:
{problem}

First draft required strictly greater than {m} {task_type} specialized roles labeling "Specialized Roles" to solve the problem collaboratively with
reasoning behind your draft of each role. Format the roles clearly, for example:
Specialized Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
m + 1. Role Name: Reasoning...
...

Then select exactly the highly {m} {task_type} influential roles labeling "Influential Roles" from the prior drafted "Specialized Roles" by re-checking the reasoning behind your selection and
assign the prior selected "Influential Roles" among exactly the {m} agents to solve the problem. Format the roles clearly, for example:
Influential Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
"""

DYNAMIC_AGENT_PROMPT = """
You are a {role}. Your task is to solve a {task_type} problem. Here is the problem that you have to
solve:
{problem}
{test_cases}

You were also given a couple of similar problems to the problem above along
with their reasoning and solutions to aid you in solving the problem at hand. Here are the similar
problems you were given:
{external_retrieval}
{self_retrieval}

And here was your original response:
{prev_response}

Also here is the leading responses with execution results from the response store:
{response_store}

Think carefully about where you went wrong, relating with responses in the response store. Then, try to
fix the solution producing a thought first then reply with a Python solution to be executed and judged again.
Make sure to wrap your code in "```python```" block, and include exactly one
block of code with the entire solution (in the final code step).
"""

VERIFIER_PROMPT = """
You are an answer extractor. Your task is to extract answer from the response to a {task_type}
problem. Here is the response for which the answer you have to extract:
{response}

Please extract the answer only inside from the "```python```" block from the response.
"""

# ========== UTILS ==========
client = openai.OpenAI(
  api_key= ''  # Replace with your actual key
)

def call_openai(messages: List[Dict[str, str]], model="gpt-4", temperature=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def extract_roles(planner_response: str, m: int) -> List[str]:
    roles = []

    # Extract everything in the Influential Roles section until next heading or end of text
    roles_section = re.search(
        r"Influential Roles:\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
        planner_response,
        flags=re.S | re.I
    )
    if roles_section:
        roles_text = roles_section.group(1).strip()
        # Extract role names before colon
        numbered_roles = re.findall(r"\d+\.\s*([^:]+):", roles_text)
        if numbered_roles:
            roles.extend([r.strip() for r in numbered_roles])

    # Deduplicate while preserving order
    seen = set()
    roles = [r for r in roles if not (r in seen or seen.add(r))]

    # If fewer than m roles, fallback to generic roles
    if len(roles) < m:
        roles.extend([f"Expert Agent {i+1}" for i in range(m - len(roles))])

    return roles[:m]

def parse_score(score_str: str) -> float:
    match = re.search(r"Score:\s*([01])", score_str, flags=re.I)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no match

def extract_python_code(candidate_response: str) -> str:
    """
    Extract the Python code inside the <python> block in the candidate response.
    """
    pattern = r"```python(.*?)```"
    match = re.search(pattern, candidate_response, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    # fallback without markdown ticks
    pattern = r"<python>\s*(.+?)\s*</python>"
    match = re.search(pattern, candidate_response, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    return ""

def run_candidate_code_and_score(code: str, test_cases: list) -> (int, str):
    """
    Run candidate code on each test case.
    Each test case is a dict with 'input' and 'output' as strings.
    Compares output and returns score and reasoning.
    """

    passed = 0
    reasoning = []

    for idx, case in enumerate(test_cases):
        input_data = case.get('input', "")
        expected_output = case.get('output', "").strip()

        # We prepare a script that:
        # - mocks input() function to provide the test case input
        # - runs the candidate code
        # - prints the result (assumes candidate code prints output)

        script = f"""
import sys
import builtins

# Mock input() to return the input_data line by line (if multiline)
input_lines = {json.dumps(input_data.splitlines())}
input_iter = iter(input_lines)
builtins.input = lambda: next(input_iter)

# Candidate code starts here
{code}
"""

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmpf:
                tmpf.write(script)
                tmpf_path = tmpf.name

            proc = subprocess.run(
                [sys.executable, tmpf_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            os.unlink(tmpf_path)

            output = proc.stdout.strip()

            if output == expected_output:
                passed += 1
                reasoning.append(f"Test case {idx+1}: Passed")
            else:
                reasoning.append(f"Test case {idx+1}: Failed\nExpected: {expected_output}\nGot: {output}")

        except Exception as e:
            reasoning.append(f"Test case {idx+1}: Exception: {e}")

    score = passed
    reasoning_str = "\n".join(reasoning)
    return score, reasoning_str

def extract_final_answer(response: str, task_type: str) -> str:
    if task_type == "coding":
          code = extract_python_code(response)
          return code
    return response.strip()

# ========== MEMORY CLASSES ==========

class EpisodicMemory:
    def __init__(self, memory_file=None):
        self.memory = []  # List of dicts: {problem, solution}
        self.tokenized_corpus = []
        self.bm25 = None
        self.memory_file = memory_file
        if memory_file:
            self.load_memory_safe(memory_file)

    def load_memory_safe(self, filepath: str):
        import os
        if os.path.exists(filepath) and os.stat(filepath).st_size > 0:
            try:
                self.load_memory(filepath)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load episodic memory from {filepath} due to JSON error: {e}. Starting empty.")
                self.memory = []
                self.tokenized_corpus = []
                self.bm25 = None
            except Exception as e:
                print(f"Warning: Error loading episodic memory from {filepath}: {e}. Starting empty.")
                self.memory = []
                self.tokenized_corpus = []
                self.bm25 = None
        else:
            # File doesn't exist or empty -> start empty
            self.memory = []
            self.tokenized_corpus = []
            self.bm25 = None

    def load_memory(self, filepath: str):
        with open(filepath, 'r') as f:
            self.memory = json.load(f)
        self.tokenized_corpus = [word_tokenize(entry['problem'].lower()) for entry in self.memory if entry.get('problem')]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, k=5) -> List[Dict[str, str]]:
      if not self.bm25:
          return []
      tokenized_query = word_tokenize(query.lower())
      scores = self.bm25.get_scores(tokenized_query)
      top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
      return [
          {"problem": self.memory[i].get("problem", ""), "solution": self.memory[i].get("solution", "")}
          for i in top_n if self.memory[i].get('solution')
      ]

    def update(self, problem: str, answer: str):
        if not problem or not answer:
            return
        self.memory.append({'problem': problem, 'solution': answer})
        self.tokenized_corpus = [word_tokenize(entry['problem'].lower()) for entry in self.memory if entry.get('problem')]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def save_memory(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)

def model_self_recall(problem: str, task_type: str) -> str:
    recall_prompt = f"""
    You are asked to recall from your own internal knowledge a relevant but distinct {task_type} problem labeling "Problem" and its solution labeling "Response",
    different from the following problem:
    {problem}

    Please provide the recalled problem and its complete solution.
    """

    messages = [{"role": "user", "content": recall_prompt}]
    recall_response = call_openai(messages)
    return recall_response.strip()

class SharedMemory:
    def __init__(self, m: int):
        self.memory = []  # List of dicts: agent, response, score
        self.m = m

    def update(self, new_entries: List[Dict]):
        self.memory.extend(new_entries)
        self.memory = sorted(self.memory, key=lambda x: x["score"], reverse=True)[:self.m]


# ========== MAIN LOOP ==========

def main():
    agents = 2
    rounds = 2
    m = agents

    episodic_memory = EpisodicMemory(memory_file="/content/episodic_memory.json")

    def read_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    questions = read_jsonl("/content/test.json")
    random.shuffle(questions)

    all_responses = []

    for data in questions[:5]:
        problem = data['question_content']
        test_cases = data['public_test_cases']
        private_test_cases = data['private_test_cases']
        task_type = "coding"
        shared_memory = SharedMemory(m)
        print(f"\n=== Solving problem ===\n{problem}\n")

        planner_prompt = PLANNER_PROMPT.format(task_type=task_type, problem=problem, m=m)
        planner_context = [{"role": "user", "content": planner_prompt}]
        planner_response = call_openai(planner_context)
        print("Planner response:\n", planner_response)

        roles = extract_roles(planner_response, m)
        if len(roles) < m:
            roles = [f"expert agent {i+1}" for i in range(m)]
        print(f"Assigned roles: {roles}")

        agent_responses = [""] * m
        judge_responses = [""] * m

        external_retrieval = episodic_memory.retrieve(problem)

        for iteration in range(rounds):
            new_entries = []

            # Build response_store from shared memory
            response_store_text = "\n".join(
                f"Agent: {entry['agent']}\nResponse: {entry['response']}\nScore: {entry['score']}\n"
                for entry in shared_memory.memory
            )
            if not response_store_text:
                response_store_text = "None"

            #print("=== Current Shared Memory ===")
            #print(response_store_text)

            for i, role in enumerate(roles):
                prev_response = agent_responses[i]

                if external_retrieval:
                    # Format external retrieval
                    external_retrieval_text = "\n\n".join(
                        f"Problem:\n{entry['problem']}\n\nResponse:\n{entry['solution']}"
                        for entry in external_retrieval
                    )
                    self_retrieval_text = "None"
                    """
                    print("\nExternal Retrieval Entries:")
                    if external_retrieval:
                        for idx, entry in enumerate(external_retrieval):
                            prob_snippet = entry['problem'][:150] + ("..." if len(entry['problem']) > 150 else "")
                            resp_snippet = entry['solution'][:300] + ("..." if len(entry['solution']) > 300 else "")
                            print(f"{idx + 1}. Problem: {prob_snippet}")
                            print(f"   Solution: {resp_snippet}\n")
                    else:
                        print("None")
                    print()
                    """
                else:
                    recall_text = model_self_recall(problem, task_type)
                    external_retrieval_text = "None"
                    self_retrieval_text = recall_text
                    """
                    print("\nSelf Retrieval Entries:")
                    problem_match = re.search(r"Problem:\s*(.*?)\n\nSolution:\s*(.*)", recall_text, re.DOTALL | re.IGNORECASE)
                    if problem_match:
                        recalled_problem = problem_match.group(1).strip()
                        recalled_response = problem_match.group(2).strip()
                    else:
                        # fallback: print full recall_text as response only
                        recalled_problem = "(unknown)"
                        recalled_response = recall_text.strip()

                    prob_snippet = recalled_problem[:150] + ("..." if len(recalled_problem) > 150 else "")
                    resp_snippet = recalled_response[:300] + ("..." if len(recalled_response) > 300 else "")
                    print(f"1. Problem: {prob_snippet}")
                    print(f"   Solution: {resp_snippet}\n")
                    """

                dynamic_prompt = DYNAMIC_AGENT_PROMPT.format(
                    role=role,
                    task_type=task_type,
                    problem=problem,
                    test_cases=test_cases,
                    external_retrieval=external_retrieval_text,
                    self_retrieval=self_retrieval_text,
                    prev_response=prev_response if prev_response else "None",
                    response_store=response_store_text
                )
                messages = [{"role": "user", "content": dynamic_prompt}]
                response = call_openai(messages)
                print(f"Agent ({role}) response (iteration {iteration}):\n{response}")

                # Parse JSON string to Python list of dicts
                parsed_test_cases = json.loads(test_cases)
                code = extract_python_code(response)
                if code:
                    score, reasoning_str = run_candidate_code_and_score(code, parsed_test_cases)
                    print(f"Judge scored agent {role} response: {score} / {len(parsed_test_cases)}")
                    print("Reasoning:\n", reasoning_str)
                    judge_responses[i] = score
                else:
                    print("No valid Python code found in the candidate response.")
                    score = 0

                agent_responses[i] = response

                shared_memory.update([{
                    "agent": role,
                    "response": response,
                    "score": score
                }])

            #parsed_test_cases = json.loads(test_cases)
            #scores = [entry['score'] for entry in shared_memory.memory]
            #if len(scores) == m and all(score == len(parsed_test_cases) for score in scores):
            #    print(f"All agents pass all test cases in iteration {iteration}. Early stopping.")
            #    break

        if shared_memory.memory:
            best_entry = max(shared_memory.memory, key=lambda x: x["score"])
        else:
            best_entry = {"response": agent_responses[0] if agent_responses else ""}

        verifier_prompt = VERIFIER_PROMPT.format(task_type=task_type, response=best_entry["response"])
        final_answer = call_openai([{"role": "user", "content": verifier_prompt}])
        print(f"Final Answer Extracted:\n{final_answer}")

        # Update episodic memory with new experience
        episodic_memory.update(problem, response)
        episodic_memory.save_memory("/content/episodic_memory.json")

        parsed_test_cases = json.loads(private_test_cases)
        code = extract_python_code(best_entry["response"])
        if code:
            score, reasoning_str = run_candidate_code_and_score(code, parsed_test_cases)
            print(f"Final test score: {score} / {len(parsed_test_cases)}")
        else:
            print("No valid Python code found in the candidate response.")
            score = 0
        if(score == len(parsed_test_cases)):
            print("Success!")
        else:
            print("Failure!")
    
        all_responses.append({
            "problem": problem,
            "planner_response":planner_response,
            "...\ndynamic_agent_responses(upon convergence)": agent_responses,
            "...\njudge_responses(upon convergence)": judge_responses,
            "shared_memory": shared_memory.memory,
            "final_answer": final_answer
        })

    # Save all responses to a pickle file
    with open("/content/all_responses.pkl", "wb") as f:
        pickle.dump(all_responses, f)

if __name__ == "__main__":
    main()
