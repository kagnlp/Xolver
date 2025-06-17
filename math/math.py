import openai
import json
import random
import re
import subprocess
import tempfile
import os
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    print("Downloading 'punkt_tab' resource for NLTK...")
    nltk.download('punkt_tab')

test = load_dataset("di-zhang-fdu/MATH500")
episodic_memory = load_dataset("nvidia/OpenMathReasoning")

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
fix the solution producing a thought later reply with a solution to be executed and judged again. You can
integrate a Python tool to execute the calculations while replying your solution if required.

Make sure to wrap your final answer in \\boxed{{answer}} block with the entire solution (in the final answer step).
"""

JUDGE_PROMPT = """
You are a judge. Your task is to judge the candidate solution of a {task_type} problem. Here is the
problem for which the candidate solution you have to judge:
{problem}

And here is the candidate response which to judge:
{candidate_response}

Please produce a score labeling "Score" (if the response is correct, it should be 1 otherwise should be 0) with reasoning
behind your judgement of the candidate solution to the problem.
"""

VERIFIER_PROMPT = """
You are an answer extractor. Your task is to extract answer from the response to a {task_type}
problem. Here is the response for which the answer you have to extract:
{response}

Please extract the answer only inside from the \\boxed{{answer}} block from the response.
"""

EQUALITY_PROMPT = """
You are a checker. Your task is to determine whether two answers to the same problem are semantically equal.

First answer:
{answer1}

Second answer:
{answer2}

If the answers are equal, output 1, otherwise 0.
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

    questions = read_jsonl("/content/test.jsonl")
    random.shuffle(questions)

    all_responses = []

    for data in questions[:1]:
        problem = data['problem']
        answer = data.get('answer', None)
        task_type = "math"
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
                    external_retrieval=external_retrieval_text,
                    self_retrieval=self_retrieval_text,
                    prev_response=prev_response if prev_response else "None",
                    response_store=response_store_text
                )
                messages = [{"role": "user", "content": dynamic_prompt}]
                response = call_openai(messages)
                print(f"Agent ({role}) response (iteration {iteration}):\n{response}")

                judge_prompt = JUDGE_PROMPT.format(task_type=task_type, problem=problem, candidate_response=response)
                score_str = call_openai([{"role": "user", "content": judge_prompt}])
                score = parse_score(score_str)
                print(f"Judge scored agent {role} response: {score} (Judge comment: {score_str[:100]}...)")

                agent_responses[i] = response
                judge_responses[i] = score_str

                shared_memory.update([{
                    "agent": role,
                    "response": response,
                    "score": score
                }])

            #scores = [entry['score'] for entry in shared_memory.memory]
            #if len(scores) == m and all(score == 1 for score in scores):
            #    print(f"All agents scored 1 in iteration {iteration}. Early stopping.")
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

        # Use LLM to compare generated answer vs ground truth
        if answer is not None:
          score_reply = call_openai([{
              "role": "user",
              "content": EQUALITY_PROMPT.format(
                  answer1=final_answer,
                  answer2=answer
              )
          }])
        is_correct = score_reply
        print(f"Ground Truth Answer: {answer}")
        print(f"Answer Match: {'Yes' if is_correct else 'No'}")

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
