# autoresearch: stage 1 hardware adaptation

Adapt the current best baseline to local single-GPU environment. Find the strongest local hyperparameter configuration for this hardware.

**File to modify**: `my_runs/2026-03-23_best_10L_base/train_gpt.py`

**Do NOT modify**: Root `train_gpt.py` (naive baseline reference)

**Ignore**: `autoresearch/` directory (reference only)

## Information Sync

- **Git**: successful experiments only (val_bpb improvements)
- **results.tsv**: all experiments (keep/discard/crash), committed to git
- **run_YYYYMMDD_HHMMSS.log**: timestamped logs, not committed

## Setup

1. **Tag**: propose tag like `mar23-local`, create branch `autoresearch/<tag>`
2. **Read**: `README.md`, `my_runs/2026-03-23_best_10L_base/README.md`, `my_runs/2026-03-23_best_10L_base/train_gpt.py` (first ~200 lines)
3. **Verify data**: `ls data/datasets/fineweb10B_sp1024/` — if missing, run `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
4. **Init results.tsv**: header row `commit	val_bpb	memory_gb	status	description` (tabs)
5. **Gitignore logs**: `grep -q "run_.*\.log" .gitignore || echo "run_*.log" >> .gitignore`
6. **Commit**: `git add my_runs/2026-03-23_best_10L_base/train_gpt.py results.tsv .gitignore && git commit -m "Stage 1: initial baseline"`
7. **Go**: confirm and start

## Scope

**CAN modify** (hardware-sensitive only):
- Batch/shape: `train_batch_tokens` (786432), `train_seq_len` (2048), `iterations` (20000)
- LRs: `matrix_lr` (0.02), `scalar_lr` (0.02), `tied_embed_lr` (0.03), `embed_lr` (0.6), `head_lr` (0.008)
- Schedule: `warmup_steps` (20), `warmdown_iters` (3000), `muon_momentum` (0.99), `muon_momentum_warmup_start` (0.92), `muon_momentum_warmup_steps` (1500)
- Regularization: `weight_decay` (0.04), `grad_clip_norm` (0.3) — use sparingly

**CANNOT modify**: architecture (num_layers, model_dim, num_heads, mlp_mult), bigram hash, SWA, evaluation, dataset, tokenizer

**Search order**: 1) batch/shape → 2) LR → 3) schedule → 4) regularization

**First run**: baseline unmodified to establish local performance

## Running Experiments

```bash
LOG_FILE=run_$(date +%Y%m%d_%H%M%S).log
RUN_ID=stage1_exp_$(date +%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 my_runs/2026-03-23_best_10L_base/train_gpt.py > $LOG_FILE 2>&1
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
phase_batch_count = 0, phase_lr_count = 0, phase_schedule_count = 0
best_val_bpb = infinity
```

**LOOP**:

1. **Check stop conditions**:
   - `experiments_no_improvement >= 15` → STOP (convergence)
   - `experiments_total >= 100` → STOP (budget)
   - All phases >= 5 experiments AND `experiments_no_improvement >= 10` → STOP (complete)

2. Pick experiment idea, note phase (batch/LR/schedule)

3. Modify `my_runs/2026-03-23_best_10L_base/train_gpt.py` (don't commit)

4. Run experiment (command above)

5. Extract results

6. Update: `experiments_total++`, increment phase counter

7. **Decision**:

   **IMPROVED** (val_bpb < best_val_bpb):
   ```bash
   git add my_runs/2026-03-23_best_10L_base/train_gpt.py
   git commit -m "Experiment: <description>"
   # Append to results.tsv: <commit>\t<val_bpb>\t<mem_gb>\tkeep\t<desc>
   git add results.tsv && git commit -m "Record kept: <desc>"
   # best_val_bpb = current, experiments_no_improvement = 0
   ```

   **NOT IMPROVED**:
   ```bash
   git checkout my_runs/2026-03-23_best_10L_base/train_gpt.py
   # Append to results.tsv: <commit>\t<val_bpb>\t<mem_gb>\tdiscard\t<desc>
   git add results.tsv && git commit -m "Record discard: <desc>"
   # experiments_no_improvement++
   ```

   **CRASHED**:
   ```bash
   git checkout my_runs/2026-03-23_best_10L_base/train_gpt.py
   # Append to results.tsv: <commit>\t0.000000\t0.0\tcrash\t<desc>
   git add results.tsv && git commit -m "Record crash: <desc>"
   # experiments_no_improvement++
   ```

8. Go to step 1

## Results Format

**results.tsv** (tab-separated, 6 decimals for val_bpb):
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.142760	22.9	keep	local baseline
a1b2c3d	1.145000	23.0	discard	increase warmup
b2c3d4e	1.140200	23.1	keep	lower batch size
```

## Stopping & Summary

When stopped, output summary:
1. **Stop reason** (convergence/complete/budget)
2. **Stats**: total experiments, keep/discard/crash counts, phase breakdown
3. **Performance**: baseline → final val_bpb, improvement %
4. **Top 3-5 changes** with impact
5. **Final config** (key hyperparams)
6. **Stage 2 suggestions** (if any)

**Review commands**:
- `cat results.tsv` — full history
- `git log --oneline` — success chain
- `head -100 my_runs/2026-03-23_best_10L_base/train_gpt.py` — current config
- `git diff <c1> <c2> my_runs/2026-03-23_best_10L_base/train_gpt.py` — compare
- `tail -100 run_20260323_143022.log` — detailed log

## Notes

- Continue autonomously, don't ask to continue unless stopped
- If low on ideas: revisit phases, try smaller adjustments, 2D sweeps
- Crashes: fix if trivial, else record and move on
- OOM → reduce batch/seq_len
- Use SAME time budget for all experiments (default 600s)
- Simpler is better — prefer robust changes over fragile tiny gains
