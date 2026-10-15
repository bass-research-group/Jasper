# GPU ANNS index evaluation: Jasper vs Tagore vs cuVS CAGRA vs ScaleGANN

This harness runs the single-GPU comparison on one machine. The multi-GPU comparison (cuVS vs ScaleGANN) will follow.

## What is measured

| | Jasper | Tagore | cuVS CAGRA | ScaleGANN |
|---|---|---|---|---|
| Build algorithm | Vamana-style incremental insert (beam search + robust prune, L=64, α=1.2), fp16 | GNN-Descent kNN (k=96) + GPU pruning (CAGRA or Vamana rule), fp16 | NN-Descent (128) + CAGRA graph optimization (64) | CPU k-means partition with selective duplication + per-shard CAGRA build + CPU merge |
| Build entry point | `create_index` CLI | `Tagore.GNN_descent` + `Tagore.Pruning` | `cuvs.neighbors.cagra.build` | `executeDiskPartition` → per-shard `cagra.build` → `executeDiskMerge` |
| Final degree | 64 | 64 | 64 | 64 |
| Native GPU search | beam search (`run_query`) | none | CAGRA search | none (CPU greedy only) |

**Build time.** Two numbers are reported for each build:
- **core**: the time the library spends building.
  - Jasper: its `[construct] timing summary` total.
  - Tagore: its `time of GNN-Descent` + `time of pruning`. Pruning also includes the D2H copy and the index file write.
  - CAGRA: the `cagra.build` call. The dataset is copied to the GPU before the timer starts.
  - ScaleGANN: partition time + the sum of the per-shard `cagra.build` times + merge time.
- **e2e**: wall-clock time of the build process(es), including data load and index save.

Peak GPU memory is sampled through NVML (peak minus the usage before launch).

**Search.** Each system's graph is exported to a fixed-degree `uint32` graph and searched with the cuVS CAGRA kernel (`cagra.from_graph` + `cagra.search`):
- The search parameters are the same for every graph, so the curves compare graph quality.
- Rows with fewer than 64 neighbours (Jasper, Tagore-Vamana) are padded by repeating that row's own neighbours.
- CAGRA search starts from random seeds, so Jasper's medoid entry point is not used in this search.

Jasper is also measured with its native beam search. That search is fp16 and is plotted against the `f16` CAGRA-kernel curves. QPS uses 10k queries in one batch (configurable), with a warmup run and the median of 5 repeats. Recall is recall@10 against exact top-100 ground truth computed on the GPU.

**ScaleGANN on one GPU.** The partitioner only creates more than one shard when the data does not fit in GPU memory (`-M`). On 1M–10M datasets the `auto` variant therefore builds one shard, which is plain CAGRA plus the partition and merge overhead. The `4shards` variant (`-N 4`) forces the out-of-memory regime that ScaleGANN is designed for: shards are built one after another on the same GPU and then merged.

ScaleGANN's original scripts build shards with RAFT-24.10 `RAFT_CAGRA_ANN_BENCH`. Here shards are built with the same cuVS `cagra.build` as the CAGRA baseline (same NN-Descent 128→64 parameters). The shard graphs are then written in the RAFT-24.10 layout that `executeDiskMerge` parses (`graphs.write_raft_cagra_graph`). Current cuVS serialization adds fields that ScaleGANN's parser does not handle.

## Constraints found in the code

- **Tagore**
  - Input must be float32 `.fvecs`.
  - `dim` must be a multiple of 32 and at most 128. `prepare_data.py` writes `base.fvecs` only when this holds; otherwise Tagore is skipped.
  - kNN `k` ≤ 96 and final degree ≤ 64.
  - Each build needs a fresh process: pruning overwrites the kNN graph in place, and the library leaks memory.
  - GPU memory is about 2.7 GB per 1M points at k=96, D=128.
  - Indexing is 32-bit, so N is limited to roughly 33M.
- **Jasper**
  - The CLI is compiled only for R ∈ {32, 64}, with L=64 fixed at compile time.
  - Storage and compute are fp16.
  - Its per-target CUDA arch lists are hardcoded to `89;90;120`. `setup.sh` overrides them through the new `-DJASPER_CUDA_ARCHS` option.
  - `setup.sh` builds into `build-eval/` in each repo and leaves your own `build/` directories alone.
  - RaBitQ code is present but not yet used by build or search.
- **ScaleGANN**
  - Its search (`searchScaleGANN`) runs on the CPU, so only its merged graph is evaluated here, on the GPU.
  - The merge keeps a random subset of the union of shard neighbours (no distance-based pruning).
  - Build it without `-DNDEBUG`: some asserts in `disk_merge.cpp` have side effects.

## Running on the evaluation machine

```bash
# 0) repos side by side, on the evaluation branch
cd ~ && for r in Jasper Tagore ScaleGANN; do
  git clone -b claude/friendly-lamport-d2qbkl git@github.com:bass-research-group/$r.git; done

# 1) environment (adjust cuda-version in env.yml to your driver)
conda env create -f ~/Jasper/eval/gpu_anns/env.yml
conda activate gpu-anns-eval

# 2) build Jasper CLI, Tagore module, ScaleGANN partition/merge
cd ~/Jasper/eval/gpu_anns
REPOS=~ ./setup.sh                       # CUDA_ARCH=<sm> to override auto-detect

# 3) data (+ exact GT on GPU). deep/bigann 10M are range-downloaded from the 1B files.
export ANNS_DATA=/path/to/data ANNS_WORK=/path/to/work
python prepare_data.py --data-root $ANNS_DATA sift-1M deep-10M bigann-10M

# 4) quick smoke test, then the full run
python run_single_gpu.py --datasets sift-1M
python run_single_gpu.py                 # all datasets/systems in configs/single_gpu.yaml

# 5) report
python report.py $ANNS_WORK/results/single_gpu.jsonl   # -> report.md + *_recall_qps.png
```

The run can be resumed: finished steps are skipped (`--force` re-runs them). Logs for each build and search go to `$ANNS_WORK/<dataset>/<system>-<variant>/*.log`. Failures, such as Tagore running out of GPU memory at 10M, are recorded with the tail of the log and the run continues.

## Files

| file | purpose |
|---|---|
| `configs/single_gpu.yaml` | datasets, degree, per-system parameters, search sweep |
| `prepare_data.py` | download/convert datasets, exact GT (cuVS brute force), Jasper GT, Tagore fvecs |
| `run_single_gpu.py` | orchestrator: build → export → search, results JSONL |
| `build.py` | per-system build/export subcommands (each runs in a fresh process) |
| `search_graph.py` | CAGRA-kernel search of any fixed-degree graph |
| `graphs.py` | readers/writers for the graph format of each system |
| `report.py` | markdown tables (build time, QPS@recall) + recall/QPS plots |
| `setup.sh`, `env.yml` | environment + builds |
