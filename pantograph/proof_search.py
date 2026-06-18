"""
Class which uses a Pantograph instance to aid proof search. All calls to the kernel uses this
interface.
"""
from pantograph.expr import Variable, Goal, GoalState, \
    Tactic, TacticHave, TacticCalc

from pantograph.server import Server, ServerError

import os
import json
import heapq
import time
import random

from datetime import datetime
from pathlib import Path
from tqdm import tqdm, trange
from openai import OpenAI

# temporarily comment out (not directly using LLM through vllm)
# import transformers
# import vllm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
    """
    Function for enabling vllm-based model to generate tactics for proof search
    """
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
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

def generate_llm_test(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
    """
    Function for testing BFS proof search algorithm manually (by inputting next tactic)
    """
    texts, scores = [], []
    for temperature in temperatures:
        new_prompt = "Using a temperature of " + str(temperature) + " and with max tokens of " + str(max_tokens) + ", answer the following question:\n" + prompt
        
        print("\033[2J\033[H", end="", flush=True)
        print(new_prompt)
        
        text = input("\nOutput: ")
        score = -random.random()

        texts.append(text)
        scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores

def generate_llm_openai_model(prompt, client, model_name, temperatures, num_samples, stop, max_tokens=256):
    """
    Function for enabling OpenAI LLM model to generate tactics for proof search
    """
    texts, scores = [], []
    for temperature in temperatures:
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_samples,
            stop=stop,
            logprobs=True
        )
        
        for i in range(num_samples):
            choice = completion.choices[i]
            
            text = ""
            score = 0.0
            token_count = 0
            
            lp_content = choice.logprobs.content
            
            for token_lp in lp_content:
                score -= token_lp.logprob
                token_count += 1
                token = token_lp.token
                
                # strip generated message to only look at first tactic generated (ignore everything after , or ; or \n)
                if ";" in token:
                    text += token_lp.token[:token.index(";")]
                    break
                elif "," in token:
                    text += token_lp.token[:token.index(",")]
                    break
                elif "\n" in token:
                    text += token_lp.token[:token.index("\n")]
                    break
                else:
                    text += token
            
            token_count = max(token_count, 1)
            score /= token_count
            
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

def _tactic_state(state):
    if isinstance(state, GoalState):
        ts = "\n".join([str(goal).strip() for goal in state.goals])
    else:
        ts = state
    return ts

def _prompt_oneshot(ts):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here is an example:

Tactic state:
---
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
---
Next tactic:
---
rw [← h.gcd_eq_one]
---

Tactic state:
---
%s
---
Next tactic:
---""" % (ts)
    return prompt

def _prompt_fewshot(ts):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

Tactic state:
---
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
---
Next tactic:
---
rintro s t ⟨u, a, hr, he⟩
---

Tactic state:
---
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
---
Next tactic:
---
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
---

Tactic state:
---
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
---
Next tactic:
---
rw [← h.gcd_eq_one]
---

Tactic state:
---
%s
---
Next tactic:
---""" % (ts)
    return prompt

def best_first_search(
        server,
        theorem,
        theorem_name,
        theorem_num,
        client,
        model_name,
        max_iters,
        temperatures,
        num_samples,
        prompt_fn,
        timeout,
        log_dir,
        early_stop=False,
        max_tokens=256
) -> dict:
    """Best first search."""
    attempt_results = []
    f = open(log_dir + "/" + theorem_name + ".txt", "a")
    
    f.write("Theorem number: " + str(theorem_count))
    
    try:
        init_state = server.goal_start(theorem)
        
        f.write("\n\nInitial state:\n")
        f.write(_tactic_state(init_state).strip())
        
        start = time.time()
        proof_finished = False
        queue = [(0.0, [], init_state, [])]
        visited = set()

        for iteration in trange(max_iters):
            # print("Iteration number: " + str(iteration+1))
            
            f.write("\n\n---------------------------")
            f.write("\nIteration " + str(iteration+1))
            f.write("\n---------------------------\n")
            
            if len(queue) == 0 or proof_finished:
                break

            total_score, steps, state, trace = heapq.heappop(queue)
            goal = state.goals[0]
            ts = _tactic_state(state)
            visited.add(ts)
            
            step_cands, step_scores = generate_llm_openai_model(
                prompt_fn(ts),
                client,
                model_name,
                temperatures,
                num_samples,
                stop='---',
                max_tokens=max_tokens
            )
                        
            step_cands = [s.strip() for s in step_cands]
            
            f.write("\nTactic candidates:\n")
            f.write(str(step_cands))
            f.write("\n\nTactic scores (-log prob):\n")
            f.write(str(step_scores))
            
            # print(step_cands)
            # print(step_scores)
            # input()
            
            step_count = 1

            for step, score in zip(step_cands, step_scores):
                result = server.goal_tactic(state, 0, step)
                step_trace = {
                    "tactic": step,
                    "state_before": ts
                }
                
                rs = _tactic_state(result)
                solved = "solved" if result.is_solved else "not solved"
                
                f.write("\n\nTactic " + str(step_count) + " result: " + solved + "\n")
                f.write(rs)
                
                # print(result)
                # print(result.is_solved)
                # input()
                
                if result.is_solved:
                    elapsed_time = time.time() - start
                    
                    attempt_results.append({
                        'theorem': theorem,
                        'proof': steps + [step],
                        'score': total_score - score,
                        'success': True,
                        'failure_reason': '',
                        'trace': trace + [step_trace],
                        'temperature': temperatures,
                        'elapsed': elapsed_time,
                        'iteration': iteration
                    })
                    if early_stop:
                        return attempt_results
                    proof_finished = True
                    
                    f.write("\n\nTheorem proven!")
                    f.write("\nElapsed time: " + str(elapsed_time))
                    break
                else:
                    if _tactic_state(result) not in visited:
                        # Score is negative log probability summed across steps
                        new_score = (total_score + score)
                        heapq.heappush(
                            queue, (new_score, steps+[step], result, trace+[step_trace])
                        )
    
    except ServerError as e:
        # print(e)
        # print(str(e))
        
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem,
                'success': False,
                'failure_reason': type(e).__name__
            })
            
            f.write("\n\nTheorem error")
            f.write("\nFailure reason: " + str(type(e).__name__) + " " + str(e))

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem,
            'success': False,
            'failure_reason': 'SearchEnded'
        })
        
        f.write("\Theorem not proven")
        f.write("Failure reason: SearchEnded")
        
    f.close()

    return attempt_results

def _save(model_name, results, args_dict, output_dir, shard):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        'results__%s__%s.json' % (model_name.replace('/', '_'), shard)
    )
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'args': args_dict
            }, f, indent=4)
        print(output_file)

def _load_model(model_name, tp_degree):
    """
    Old function to load model using vllm
    """
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype='bfloat16',
        max_num_batched_tokens=4096
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def _load_client(api_key):
    """
    New function to load client using OpenAI API
    """
    client = OpenAI(
        api_key=api_key
    )
    return client

def _load_data(dataset_name, dataset_path):
    """
    Old function to load dataset from minif2f dataset
    """
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        # repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
        repo = None
    else:
        raise NotImplementedError(dataset_name)

    return repo, data

def _load_data2(dataset_name, dataset_path):
    """
    New function load dataset from test dataset (only meant to check functionality of proof search)
    """
    data = []
    with open(dataset_path) as f:
        for line in f.readlines():
            data_ = json.loads(line)
            data.append(data_)
    
    return data

def _get_api_key(path):
    """
    Function to get OpenAI API key stored externally (would be normally stored in the program in another method)
    """
    key_file = open(path, 'r')
    key = key_file.readlines()[0]
    key_file.close()
    return key

def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")

def resume_from(results_filename, data):
    results = json.load(open(results_filename))['results']
    data = data[len(results):]
    print("=== Resuming from %d" % (len(results)))
    return results, data

def make_output_dir(output_dir):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return output_dir, log_dir

def pause():
    """
    Infinite while loop for debugging purposes
    """
    while True:
        x = 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name', 
        choices=[
            'open-web-math/llemma_7b',
            'open-web-math/llemma_34b',
            'codellama/CodeLlama-7b-hf',
            'codellama/CodeLlama-34b-hf',
            'gpt-3.5-turbo',
            'gpt-4'
        ],
        default='gpt-3.5-turbo'
    )
    parser.add_argument(
        '--dataset-name',
        choices=['minif2f-valid', 'minif2f-test', 'test-set'],
        default='test-set'
    )
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument(
        '--dataset-path',
        choices=['data/minif2f.jsonl', 'data/test.jsonl'],
        default='data/test.jsonl'
    )
    parser.add_argument('--output-dir', default='output/test')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--max-iters', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--num-examples', type=int, default=-1)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--clear-process-hours', type=int, default=3)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.0])
    parser.add_argument('--api-key-path', type=str, default='../api_key/key.txt')
    args = parser.parse_args()

    key = _get_api_key(args.api_key_path)
    client = _load_client(key)
    model_name = args.model_name

    output_dir, log_dir = make_output_dir(args.output_dir)
    
    repo = None
    data = None
    shard_size = None
    
    if (args.dataset_name == 'test-set'):
        data = _load_data2(args.dataset_name, args.dataset_path)
    else:
        repo, data = _load_data(args.dataset_name, args.dataset_path)
        shard_size = len(data) // args.num_shards
        data = data[args.shard*shard_size:(args.shard+1)*shard_size]
        print("Shard size: %d" % (len(data)))

    if args.resume_from is not None:
        results, data = resume_from(args.resume_from, data)
    else:
        results = []

    start = time.time()
    server = Server()
    
    theorem_count = 1
    
    for example in tqdm(data, total=len(data)):
        
        file_path = example['file_path']
        theorem_name = example['full_name']
        theorem = example['statement']
        
        attempt_results = best_first_search(
            server=server,
            theorem=theorem,
            theorem_name=theorem_name,
            theorem_num=theorem_count,
            client=client,
            model_name=model_name,
            max_iters=args.max_iters,
            temperatures=args.temperatures,
            num_samples=args.num_samples,
            prompt_fn=_prompt_oneshot,
            timeout=args.timeout,
            log_dir=log_dir,
            early_stop=args.early_stop
        )
        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }
        results.append(result)
        
        theorem_count += 1

        _save(
            model_name=args.model_name,
            results=results,
            args_dict=args.__dict__,
            output_dir=output_dir,
            shard=args.shard
        )
        print_stats(results)
        # The proof search occasionally leaves Lean processes open. As a workaround,
        # we periodically kill all Lean processes. Note that this may cause a proof search failure.
        if args.shard == 0:
            hours = 60*60*args.clear_process_hours
            if time.time() - start > hours:
                print("=== Killing active leanprover processes to mitigate leak")
                os.system("ps aux | grep leanprover | awk '{print $2}' | xargs kill -9")
                start = time.time()