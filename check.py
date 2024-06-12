import json
import heapq
import subprocess
import os
import time
import transformers
import vllm
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm, trange

from repl_wrapper import InteractiveThread

os.environ["TOKENIZERS_PARALLELISM"] = "true"  #remove warnings

def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
    else:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

    return data

def generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens=256, stop=["\ntheorem","\n\n\n", "---", "[/TAC]"]):
    texts, scores = [], []
    params = vllm.SamplingParams(
        n=num_samples,
        temperature=temperature,
        use_beam_search=(temperature==0.0 and num_samples>1),
        max_tokens=max_tokens,
        stop=stop,
    )
    outputs = model.generate([prompt], params, use_tqdm=False)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
        score = output.cumulative_logprob/max(len(output.token_ids), 1)
        texts.append(text)
        scores.append(score)
    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _prompt_fewshot(theorem_statement, state = None, tactics = None, task = "full_proof"):
    with open(f"prompt/{task}.txt", "r") as infile:
        prompt = infile.read()
    
    if task == "full_proof":
        prompt += theorem_statement + " := by"
        return prompt 
    elif task == "tactic_prediction":
        ctx = theorem_statement + "\n  " + "  ".join(tactics)
        prompt = prompt.format(state)
        return prompt
    elif task == "state_prediction":
        prompt += theorem_statement
        return prompt

def _load_model(model_name, tp_degree):
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype='bfloat16',
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def _error_message(output):
    if output == None: return True
    if "messages" in output:
        for d in output["messages"]:
            return True
    if "message" in output:
        if "error" in output["message"]:
            return True
    return False

def get_goal(output):
    if "sorries" in output and len(output["sorries"])>0:
        if 'goals' in output['sorries'][0] and len(outputs['sorries'][0]['goals'])>0:
            return output['sorries'][0]['goals'][0]
        if 'goal' in output['sorries'][0]:
            return output['sorries'][0]['goal']
    if 'goals' in output and len(output['goals']) > 0:
        return output['goals'][0]

    return None

def evaluation(example, task, thread_id, repl_path, lean_env_path, model, tokenizer, prompt_fn, temperature, num_samples, max_tokens=512):
    theorem_statement = example["statement"] + " := by"

    if task == "full_proof":
        prompt = prompt_fn(theorem_statement, '', task)
        responses = generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens)
        
        thread = InteractiveThread(thread_id, repl_path, lean_env_path)
        thread.start()

        imports = "import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology\n"
        cmd = {"cmd": imports + theorem_statement + responses[0][0]}
        output = thread.submit_and_receive(cmd)
        thread.stop()
        thread.join()

        if len(output) == 1 and 'env' in output: 
            return "Correct", output
        return "Error", output

        thread.stop()
        thread.join()
        return "Error" 

def _eval_tactic(next_tactic, thread, proof_state, example):
    if 'admit' in next_tactic or 'sorry' in next_tactic:
        return {"status": "invalid"}
    output = thread.submit_and_receive({"tactic": next_tactic, "proofState": proof_state})
    if not _error_message(output):
        if output == "" and "sorry" not in output:
            return {"status": "invalid"}
        elif example["full_name"] != None and example["full_name"] in next_tactic:
        # FIXME: forbid recursion
            return {"status": "invalid"}
        elif output["goals"] == [] and len(output) == 2: 
            return {"status": "done", "output": output}
        elif output["proofState"] > proof_state:
            return {"status": "valid", "output": output}
        else:
            return {"status": "invalid"}
    return {"status": "invalid"}


def best_first_search(example, task, thread_id, repl_path, lean_env_path, model, tokenizer, prompt_fn, temperature, num_samples, max_tokens=64, max_iters=250):
    theorem_statement = example["theoremStatement"] + " := by"
    imports = example["srcContext"]
    
    thread = InteractiveThread(thread_id, repl_path, lean_env_path, initial_context = imports, timeout=600)
    thread.start()
    output = thread.submit_and_receive({"cmd": theorem_statement + " sorry", "env": 0})

    if output != None and "sorries" in output:
        proof_state = output["sorries"][-1]["proofState"]
    else:
        thread.stop()
        thread.join()
        return {'success': False, 'msg': "Search ended"}
    goal = get_goal(output)

    print()
    print(f"Problem: {theorem_statement}")
    if goal is None:
        thread.stop()
        thread.join()
        return {
            'success': False, 
            'msg': output
        }

    # Score, steps-so-far, goal state, proof state ID
    queue = [(0.0, [], goal, proof_state)]

    for iteration in trange(max_iters):
        if len(queue) == 0:
            break
        # Dequeue the tuple with minimum score
        score, tactics, goal, proof_state = heapq.heappop(queue)

        prompt = prompt_fn(theorem_statement, goal, tactics, task)
        tactic_cands, tactic_scores = generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens)
        visited = set([goal])
        for next_tactic, tactic_score in zip(tactic_cands, tactic_scores):
            outcome = _eval_tactic(next_tactic, thread, proof_state, example)
            if outcome['status'] == 'done':
                thread.stop()
                thread.join()
                return {'success': True, 'proof': tactics + [next_tactic]}
            elif outcome['status'] == 'valid':
                new_goal = get_goal(outcome['output'])
                new_proof_state = outcome['output']['proofState']
                if new_goal not in visited:
                    heapq.heappush(queue, (
                        (score - tactic_score), 
                        tactics + [next_tactic], 
                        new_goal, 
                        new_proof_state
                    ))
                    visited.add(new_goal)
            else:
                continue

    thread.stop()
    thread.join()
    return {'success': False, 'msg': "Search ended"}

def _print_output(example, out):
    print(out)
    if 'proof' in out:
        print(example['theoremStatement'] + ' := by\n  ' + '  '.join(out['proof']))

def normalize_string(s):
    return '\n'.join([line.replace(' ', '') for line in s.strip().split('\n')])

def compare_lean_state(correct_state, output_string):
    start_index = output_string.find("/-")
    comment_content = ""
    if start_index >= 0:
        output_string = output_string[start_index+2:]
        print(output_string)
        end_index = output_string.find("-/")
        if end_index > 0: comment_content = output_string[:end_index]
        
    normalized_correct = normalize_string(correct_state)
    normalized_output = normalize_string(comment_content)

    correct_lines = normalized_correct.split('\n')[1:-1]
    output_lines = normalized_output.split('\n')
    
    matching_lines_set = set(correct_lines).intersection(set(output_lines))
    matching_lines_count = len(matching_lines_set)
    
    total_lines = len(correct_lines)
    percentage_matched = (matching_lines_count / total_lines) if total_lines > 0 else 0
    
    return percentage_matched

def state_prediction(example, task, thread_id, repl_path, lean_env_path, model, tokenizer, prompt_fn, temperature, num_samples, max_tokens=512):
    theorem_statement = example["query"]
    prompt = prompt_fn(theorem_statement, task = task)
    responses = generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens)
    print(max([compare_lean_state(example["response"], response) for response in responses[0]]))
    # return max([compare_lean_state(example["response"], response) for response in responses])


def make_output_dir(output_dir):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def get_full_name(statement):
    word_list = statement.split()
    for i in range(len(word_list)):
        if "theorem" in word_list[i] or "lemma" in word_list[i]:
            return statement.split()[i+1]
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument(
        '--task',
        default='tactic_prediction',
        choices=['tactic_prediction', 'full_proof', 'state_prediction']
    )
    parser.add_argument(
        '--dataset-name',
        default='minif2f-test'
    )
    parser.add_argument('--dataset-path', default='data/minif2f.jsonl')
    parser.add_argument('--output-dir', default='output/minif2f')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--temperatures', type=float, default=0.0)
    parser.add_argument('--repl-path', default='./repl')
    parser.add_argument('--lean-env-path', default='./miniF2F-lean4')

    
    args = parser.parse_args()
    model, tokenizer = _load_model(args.model_name, args.tp_degree)

    output_dir = make_output_dir(args.output_dir)
    examples = _load_data(args.dataset_name, args.dataset_path)
    
    prompt_fn = _prompt_fewshot

    if args.task == 'tactic_prediction':
        evaluation = best_first_search

    # FIXME: implement parallel
    thread_id = 1

    successes = 0
    count = 0
    for example in examples:
        example["full_name"] = get_full_name(example["theoremStatement"])
        count += 1
        out = evaluation(example, args.task, thread_id, args.repl_path, args.lean_env_path, model, tokenizer, prompt_fn, args.temperatures, args.num_samples, max_iters = args.max_iters)
        if out['success']: 
            successes += 1
            example["proof"] = (True, out["proof"])
        else:
            example["proof"] = (False, None)
        _print_output(example, out)
        print(f"Successes: {successes}/{count} ")
    
    filename = 'result.jsonl'

    filepath = output_dir + filename

    with open(filepath, 'w') as file:
        for entry in examples:
            json.dump(entry, file)
            file.write('\n')