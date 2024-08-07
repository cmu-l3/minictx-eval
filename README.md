# minictx-eval
## repl_wrapper.py
The wrapper class of interactive lean repl. 
Create a new thread by calling ```InteractiveThread(thread_id, repl_path, lean_env_path)```, where 
- thread_id: any number
- repl_path: the path to repl directory
- lean_env_path: the path to the lean project containing environment you want to test

```
thread = InteractiveThread(1, repl_path, lean_env_path)
thread.start()

cmd = {'cmd': 'import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology'}
output = thread.submit_and_receive(cmd)

thread.close()
thread.join()
```
```thread.submit_and_receive``` take a dictionary as input and return the output of repl in dictionary


## check.py (In progress)
```evaluation(example, task, thread_id, repl_path, lean_env_path, model, tokenizer, prompt_fn, temperature, num_samples, max_tokens=512)``` supports evaluate on a single problem formulated as ```example``` dictionary. It supports both full proof and tactic prediction
