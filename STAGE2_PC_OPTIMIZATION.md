# autoresearch: stage 2 PC layer optimization

Optimize the PC (Preconditioned) Layer to achieve further performance improvements beyond stage 1 baseline. PC Layer has been integrated into `train_gpt.py`.

**File to modify**: `train_gpt.py` (in the root directory)

**Goal**: Use PC Layer as a new method to bring additional val_bpb improvements.

## Challenge Rules

- **Training time**: 10 minutes
- **Objective**: Minimize val_bpb (bits per byte) on FineWeb validation set
- **Artifact size**: Model + code must compress to <16MB (zstd-22)
- **Evaluation**: No training data access during eval; eval time <10 minutes

## Information Sync

- **Git**: successful experiments only (val_bpb improvements)
- **results.tsv**: all experiments (keep/discard/crash), committed to git
- **run_YYYYMMDD_HHMMSS.log**: timestamped logs, not committed

## Setup

1. **Branch**: propose tag like `pc-opt`, create from current branch: `git checkout -b autoresearch/<tag>`
2. **Read**: `train_gpt.py` (understand PC Layer hyperparameters and implementation)
3. **Verify stage 1**: ensure stage 1 optimization is complete and baseline is solid
4. **Init results.tsv**: if not exists, create with header `commit	val_bpb	memory_gb	status	description` (tabs) in the root dir.
5. **Gitignore logs**: `grep -q "run_.*\.log" .gitignore || echo "run_*.log" >> .gitignore`
6. **Commit**: `git add train_gpt.py results.tsv .gitignore && git commit -m "Stage 2: PC layer optimization start"`
7. **Go**: confirm and start

## Scope

**Primary focus**: PC Layer hyperparameters and methodology

**Good starting point** (recommended first experiment):
- `pc_level=2`
- `pc_norm_type='F'`
- `PC_DETACH_NORM_COMPUTE=0` (not detach)
- `PC_LEARNABLE_GAMMA=1` (use learnable gamma)
- Other PC parameters remain default

**CAN modify**:
- PC hyperparameters: `pc_level`, `pc_norm_type`, `PC_DETACH_NORM_COMPUTE`, `PC_LEARNABLE_GAMMA`, and any other PC-related parameters
- PC Layer implementation itself (if hyperparameter tuning alone doesn't bring improvements)
- Stage 1 hyperparameters: batch/shape, LRs, schedule, regularization (can be re-tuned with PC Layer)
- Interaction with optimizer/scheduler if needed for PC Layer

**CANNOT modify**: architecture (num_layers, model_dim, num_heads, mlp_mult), bigram hash, SWA, evaluation, dataset, tokenizer

**Search strategy**:
1. Start with recommended PC hyperparameters
2. Tune PC hyperparameters systematically
3. If hyperparameter tuning plateaus, consider PC Layer methodology modifications

**First run**: recommended starting point to establish PC Layer baseline

## Running Experiments

**Activate Env**:
```bash
source /zhaokunxiang/miniconda3/bin/activate base
```

**Run Experiment**:
```bash
LOG_FILE=run_$(date +%Y%m%d_%H%M%S).log
RUN_ID=stage2_exp_$(date +%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > $LOG_FILE 2>&1
```

**Extract results**:
```bash
LAST_LOG=$(ls -t run_*.log 2>/dev/null | head -1)
grep "val_bpb:\|peak_memory_mb:" $LAST_LOG
```

**Crash debug**: `tail -n 100 $LAST_LOG`

**Memory calc**: `memory_gb = peak_memory_mb / 1024`

## Experiment Loop

**Init counters**:
```
experiments_total = 0
experiments_no_improvement = 0
best_val_bpb = infinity (or stage 1 final val_bpb)
```

**LOOP**:

1. **Check stop conditions**:
   - `experiments_no_improvement >= 15` → STOP (convergence)
   - `experiments_total >= 100` → STOP (budget)

2. Pick experiment idea (PC hyperparameter or methodology change)

3. Modify `train_gpt.py` (don't commit)

4. Run experiment (command above)

5. Extract results

6. Update: `experiments_total++`

7. **Decision**:

   **IMPROVED** (val_bpb < best_val_bpb):
   ```bash
   git add train_gpt.py
   git commit -m "PC experiment: <description>"
   # Append to results.tsv: <commit>\t<val_bpb>\t<mem_gb>\tkeep\t<desc>
   git add results.tsv && git commit -m "Record kept: <desc>"
   # best_val_bpb = current, experiments_no_improvement = 0
   ```

   **NOT IMPROVED**:
   ```bash
   git checkout train_gpt.py
   # Append to results.tsv: <commit>\t<val_bpb>\t<mem_gb>\tdiscard\t<desc>
   git add results.tsv && git commit -m "Record discard: <desc>"
   # experiments_no_improvement++
   ```

   **CRASHED**:
   ```bash
   git checkout train_gpt.py
   # Append to results.tsv: <commit>\t0.000000\t0.0\tcrash\t<desc>
   git add results.tsv && git commit -m "Record crash: <desc>"
   # experiments_no_improvement++
   ```

8. Go to step 1

## Results Format

**results.tsv** (tab-separated, 6 decimals for val_bpb):
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.135000	23.2	keep	PC baseline: level=2, norm=F, no_detach, learnable_gamma
a1b2c3d	1.136500	23.3	discard	PC level=3
b2c3d4e	1.133800	23.1	keep	PC norm_type=2
```

## Stopping & Summary

When stopped, output summary:
1. **Stop reason** (convergence/budget)
2. **Stats**: total experiments, keep/discard/crash counts
3. **Performance**: stage 1 final → stage 2 final val_bpb, improvement %
4. **Top 3-5 PC changes** with impact
5. **Final PC config** (key PC hyperparameters)
6. **Insights**: what worked, what didn't, PC Layer effectiveness

**Review commands**:
- `cat results.tsv` — full history
- `git log --oneline` — success chain
- `head -200 train_gpt.py` — current config
- `git diff <c1> <c2> train_gpt.py` — compare
- `tail -100 run_20260328_143022.log` — detailed log

## Notes

- Continue autonomously, don't ask to continue unless stopped
- Record ALL attempts in results.tsv (keep/discard/crash)
- Only commit successful cases (val_bpb improvements)
- If PC hyperparameter tuning plateaus, consider modifying PC Layer implementation
- Crashes: fix if trivial, else record and move on
- OOM → adjust PC parameters or reduce complexity
- Use SAME time budget for all experiments
- Focus on robust improvements, not fragile gains
