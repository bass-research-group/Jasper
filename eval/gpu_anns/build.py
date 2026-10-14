#!/usr/bin/env python3
"""Per-system build steps. Each subcommand runs in a fresh process (launched by
run_single_gpu.py), writes a fixed-degree ``graph.u32bin`` and prints one
``@@RESULT {...}`` line with its timings.

Subcommands
-----------
cagra            cuVS CAGRA build (graph only) from base file
tagore           Tagore GNN-Descent + pruning (CAGRA / Vamana / NSG pruning)
scalegann-shards build a CAGRA graph for every ScaleGANN partition and store it
                 in the RAFT layout read by ScaleGANN's executeDiskMerge
export-jasper    convert a Jasper index file to graph.u32bin
export-scalegann convert a ScaleGANN merged index to graph.u32bin
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from common import emit_result, graph_stats, read_bin, write_bin
import graphs


def _sync():
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()


def cagra_build(data, degree, inter_degree, algo, nn_descent_iters=None):
    """Build a CAGRA graph from a host array. Returns (graph, build_s, h2d_s).
    The dataset is copied to the device before the timer starts so that the
    build time excludes I/O, like the Jasper and Tagore timers."""
    import cupy as cp
    from cuvs.neighbors import cagra

    t0 = time.perf_counter()
    d_data = cp.asarray(data)
    _sync()
    h2d = time.perf_counter() - t0

    kw = dict(metric="sqeuclidean", graph_degree=degree, intermediate_graph_degree=inter_degree,
              build_algo=algo)
    if nn_descent_iters and algo == "nn_descent":
        kw["nn_descent_niter"] = nn_descent_iters
    params = cagra.IndexParams(**kw)
    t0 = time.perf_counter()
    index = cagra.build(params, d_data)
    _sync()
    build_s = time.perf_counter() - t0
    g = graphs.cagra_index_graph(index)
    del index, d_data
    return g, build_s, h2d


def cmd_cagra(a):
    t0 = time.perf_counter()
    data = read_bin(a.base)
    load_s = time.perf_counter() - t0
    g, build_s, h2d = cagra_build(data, a.degree, a.inter_degree, a.algo, a.nn_descent_iters)
    write_bin(a.out, g)
    emit_result(dict(build_core_s=build_s, stages=dict(load_s=load_s, h2d_s=h2d, build_s=build_s),
                     graph=graph_stats(g)))


def cmd_tagore(a):
    sys.path.insert(0, str(Path(a.tagore_build).resolve()))
    import Tagore  # noqa: E402  (pybind module from Tagore/build)

    n, dim = a.n, a.dim
    raw = Path(a.raw_out)
    t0 = time.perf_counter()
    ptrs = Tagore.GNN_descent(a.knn_k, n, dim, a.iters, str(a.fvecs))
    t_knn = time.perf_counter() - t0  # includes fvecs load + fp16 convert + H2D
    t0 = time.perf_counter()
    # Positional: (K, POINTS, DIM, FINAL_DEGREE, TOPM, ptrs, index_type, thre, index_path)
    Tagore.Pruning(a.knn_k, n, dim, a.degree, a.m, ptrs, a.prune, a.alpha, str(raw))
    t_prune = time.perf_counter() - t0  # includes D2H + writing the index file
    sys.stdout.flush()

    if a.prune == "CAGRA":
        g, nvalid = graphs.read_tagore_cagra(raw, n)
    elif a.prune in ("Vamana", "NSG"):
        if a.prune == "NSG":
            raise SystemExit("NSG export not implemented (NSG header differs from Vamana)")
        g, nvalid, _ = graphs.read_diskann_mem_graph(raw, n)
    else:
        raise SystemExit(f"unsupported prune type {a.prune}")
    write_bin(a.out, g)
    if not a.keep_raw:
        raw.unlink()
    # The library's own timers ("time of GNN-Descent" / "time of pruning") are
    # parsed from stdout by the orchestrator; here we report Python wall clocks.
    emit_result(dict(stages=dict(gnn_descent_wall_s=t_knn, pruning_wall_s=t_prune),
                     graph=graph_stats(g, nvalid)))


def cmd_scalegann_shards(a):
    base = Path(a.base_folder)
    parts = sorted((p for p in base.iterdir() if p.is_dir() and p.name.startswith("partition")),
                   key=lambda p: int(p.name[len("partition"):]))
    shard_build_s, shard_wall_s, sizes = [], [], []
    for p in parts:
        data_file = next(f for f in p.iterdir() if f.name.startswith("data."))
        t0 = time.perf_counter()
        data = read_bin(data_file)
        g, build_s, _ = cagra_build(data, a.degree, a.inter_degree, a.algo, a.nn_descent_iters)
        (p / "index").mkdir(exist_ok=True)
        graphs.write_raft_cagra_graph(p / "index" / a.index_name, g, dim=data.shape[1])
        wall = time.perf_counter() - t0
        shard_build_s.append(build_s)
        shard_wall_s.append(wall)
        sizes.append(int(data.shape[0]))
        print(f"  shard {p.name}: n={data.shape[0]} build={build_s:.2f}s wall={wall:.2f}s", flush=True)
    emit_result(dict(n_shards=len(parts), shard_sizes=sizes, shard_build_s=shard_build_s,
                     shard_wall_s=shard_wall_s, shards_build_sum_s=float(sum(shard_build_s)),
                     shards_wall_sum_s=float(sum(shard_wall_s))))


def cmd_export_jasper(a):
    g, nvalid, medoid = graphs.read_jasper_graph(a.index, a.dim, a.R)
    write_bin(a.out, g)
    emit_result(dict(graph=graph_stats(g, nvalid), medoid=medoid))


def cmd_export_scalegann(a):
    g = graphs.read_npy_stream_graph(a.index)
    write_bin(a.out, g.astype(np.uint32))
    emit_result(dict(graph=graph_stats(g)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cagra_args(p):
        p.add_argument("--degree", type=int, default=64)
        p.add_argument("--inter-degree", type=int, default=128)
        p.add_argument("--algo", default="nn_descent", choices=["nn_descent", "ivf_pq", "iterative_cagra_search"])
        p.add_argument("--nn-descent-iters", type=int, default=None)

    p = sub.add_parser("cagra")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    cagra_args(p)

    p = sub.add_parser("tagore")
    p.add_argument("--tagore-build", required=True, help="dir containing the Tagore*.so module")
    p.add_argument("--fvecs", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--knn-k", type=int, default=96)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--degree", type=int, default=64)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--prune", default="CAGRA", choices=["CAGRA", "Vamana", "NSG"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--raw-out", required=True)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--out", required=True)

    p = sub.add_parser("scalegann-shards")
    p.add_argument("--base-folder", required=True)
    p.add_argument("--index-name", required=True)
    cagra_args(p)

    p = sub.add_parser("export-jasper")
    p.add_argument("--index", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--out", required=True)

    p = sub.add_parser("export-scalegann")
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)

    a = ap.parse_args()
    {"cagra": cmd_cagra, "tagore": cmd_tagore, "scalegann-shards": cmd_scalegann_shards,
     "export-jasper": cmd_export_jasper, "export-scalegann": cmd_export_scalegann}[a.cmd](a)


if __name__ == "__main__":
    main()
