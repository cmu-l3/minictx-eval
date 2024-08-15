# minictx-eval

This repository contains the evaluation scripts for miniCTX: Neural Theorem Proving with (Long-)Contexts.

## Requirements

- Python 3
- PyTorch
- Required Python packages (specified in `requirements.txt`)

  ```bash
  pip install -r requirements.txt
  ```

- Lean 4
- [Mathlib 4](https://github.com/leanprover-community/mathlib4), [HTPI](https://github.com/your-htpi-repo-link), [PrimeNumberTheoremAnd](https://github.com/your-prime-number-theorem-and-repo-link), [PFR](https://github.com/your-pfr-repo-link), or any other Lean project to test
- [Lean REPL](https://github.com/leanprover-community/repl)

## Setup Environment

1. **Install Lean 4**

   Follow the instructions on the [Lean 4 installation page](https://leanprover.github.io/lean4/doc/quickstart.html) to set up Lean 4.

2. **Set up and build your target Lean project**

   To setup the projects in this repository:
   ```bash
   git submodule init
   git submodule update
   ```

   Then build the project; for instance, for Mathlib:
   ```bash
   cd mathlib4
   lake exe cache get
   lake build
   ```

2. **Set up and build Lean REPL**

   After running `git submodule init` and `git submodule update`:
   
   ```bash
   cd repl
   lake build
   ```

## Running the Evaluation

### Edit the Script

Open `script.sh` and verify that the paths and parameters are correctly set according to your setup. The script contains the following variables:

- `TASK`: The task name, selected from `tactic_prediction`, `tactic_prediction_context`, `full_proof`, `full_proof_context`.
- `MAX_ITERS`: The maximum number of iterations (default: `100`).
- `NUM_SAMPLES`: The number of samples (default: `32`).
- `TEMPERATURES`: The temperatures for sampling (default: `0.0`).
- `DATASET`: The name of the dataset (default: `mathlib`).
- `DATA`: The path to the dataset file (default: `data/mathlib.jsonl`).
- `MODEL`: The model name (default: `l3lab/ntp-mathlib-context-deepseek-coder-1.3b`).
- `NAME`: The name for the output directory (default: `deepseekCT`).
- `OUTPUT_DIR`: The directory where the output will be saved.
- `REPL_PATH`: The path to the REPL project.
- `LEAN_PATH`: The path to the Lean project environment (e.g., Mathlib 4).

### Run the Script

Make the script executable and run it:

```bash
chmod +x script.sh
./script.sh
```

### Output

The output will be saved in the `output/${NAME}_mathlib` directory, and the log will be saved in `mathlib_context.out`.

## `repl_wrapper.py`

The evaluation code interacts with the Lean compiler using the Lean REPL. `repl_wrapper.py` provides a Python interface to interact with the Lean REPL directly.

### Usage

Create a new thread by calling `InteractiveThread(thread_id, repl_path, lean_env_path)`, where:

- `thread_id`: Any number
- `repl_path`: The path to the REPL directory
- `lean_env_path`: The path to the Lean project containing the environment you want to test

Example:

```python
from repl_wrapper import InteractiveThread

thread = InteractiveThread(1, repl_path, lean_env_path)
thread.start()

cmd = {'cmd': 'import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology'}
output = thread.submit_and_receive(cmd)

thread.close()
thread.join()
```

`thread.submit_and_receive` takes a dictionary as input and returns the output of the REPL in a dictionary.
