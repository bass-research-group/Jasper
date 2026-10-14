#!/usr/bin/env python3
"""Search any fixed-degree graph with the cuVS CAGRA search kernel.

Loads ``graph.u32bin`` + the base vectors through ``cagra.from_graph`` and
sweeps (itopk_size, search_width). This isolates *graph quality*: Jasper,
Tagore, ScaleGANN and CAGRA graphs are all searched by the same kernel with
the same parameters. QPS is measured with all queries split into batches of
``--batch-size``; each sweep point is warmed up once and timed ``--repeats``
times (median reported). Emits one ``@@RESULT`` line with a list of points.
"""

import argparse
import itertools
import time

import numpy as np

from common import emit_result, read_bin, recall_at_k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--itopk", type=int, nargs="+", default=[32, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--search-width", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", default="native", choices=["native", "f16"],
                    help="f16 converts base+queries to float16 (Jasper/Tagore compute in fp16)")
    ap.add_argument("--algo", default="auto", choices=["auto", "single_cta", "multi_cta", "multi_kernel"])
    a = ap.parse_args()

    import cupy as cp
    from cuvs.common import Resources
    from cuvs.neighbors import cagra

    base = read_bin(a.base)
    queries = read_bin(a.query)
    gt = read_bin(a.gt)
    graph = read_bin(a.graph)
    assert graph.shape[0] == base.shape[0], f"graph rows {graph.shape[0]} != base rows {base.shape[0]}"
    if a.dtype == "f16":
        if base.dtype != np.float32:
            raise SystemExit("f16 search only applies to float datasets")
        base, queries = base.astype(np.float16), queries.astype(np.float16)

    res = Resources()
    d_base = cp.asarray(base)
    d_graph = cp.asarray(graph)
    index = cagra.from_graph(d_graph, d_base, metric="sqeuclidean", resources=res)
    res.sync()

    nq = queries.shape[0]
    bs = min(a.batch_size, nq)
    batches = [cp.asarray(queries[s:s + bs]) for s in range(0, nq, bs)]
    outs = [[cp.empty((b.shape[0], a.k), dtype=cp.float32), cp.empty((b.shape[0], a.k), dtype=cp.uint32)]
            for b in batches]
    prealloc = [True]
    points = []
    for itopk, sw in itertools.product(a.itopk, a.search_width):
        if itopk < a.k:
            continue
        sp = cagra.SearchParams(itopk_size=itopk, search_width=sw, algo=a.algo)

        def run():
            for q, o in zip(batches, outs):
                if prealloc[0]:
                    try:
                        cagra.search(sp, index, q, a.k, neighbors=o[1], distances=o[0], resources=res)
                        continue
                    except (TypeError, ValueError):
                        prealloc[0] = False  # older cuvs: different neighbor dtype; let it allocate
                o[0], o[1] = cagra.search(sp, index, q, a.k, resources=res)
            res.sync()

        try:
            run()  # warmup (also JIT / plan setup)
        except Exception as e:
            print(f"  itopk={itopk} sw={sw}: {e}")
            continue
        times = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        found = np.concatenate([cp.asnumpy(cp.asarray(o[1])) for o in outs])
        rec = recall_at_k(found, gt, a.k)
        qps = nq / t
        points.append(dict(itopk=itopk, search_width=sw, qps=qps, recall=rec, time_s=t))
        print(f"  itopk={itopk:4d} sw={sw:2d}  recall@{a.k}={rec:.4f}  QPS={qps:,.0f}", flush=True)
    emit_result(dict(points=points, batch_size=bs, k=a.k, dtype=a.dtype, n_queries=nq))


if __name__ == "__main__":
    main()
