#!/usr/bin/env python3
"""Single-GPU benchmark driver: Jasper vs Tagore vs cuVS CAGRA vs ScaleGANN.

For every dataset x system(-variant):
  1. build the index in a fresh process (timed, peak GPU memory sampled),
  2. export the graph as a fixed-degree ``graph.u32bin``,
  3. search it with the cuVS CAGRA kernel (``search_graph.py``) -- same kernel
     for every graph, so differences are graph quality,
  4. for Jasper, also run its native beam search (``run_query``).

Results are appended to ``<work_root>/results/single_gpu.jsonl``; finished
steps are skipped on re-run (``--force`` to redo). Use ``report.py`` to plot.

    python run_single_gpu.py -c configs/single_gpu.yaml
    python run_single_gpu.py -c configs/single_gpu.yaml --datasets sift-1M --systems cagra jasper
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common import (GpuMemMonitor, append_jsonl, gpu_info, load_jsonl, parse_emitted, parse_float,
                    run_logged)

HERE = Path(__file__).resolve().parent
PY = sys.executable
SYSTEMS = ["cagra", "jasper", "tagore", "scalegann"]


def expand(p):
    return Path(os.path.expandvars(os.path.expanduser(str(p)))) if p is not None else None


class Runner:
    def __init__(self, cfg, force=False):
        self.cfg = cfg
        self.force = force
        self.data_root = expand(cfg["data_root"])
        self.work_root = expand(cfg["work_root"])
        self.results = self.work_root / "results" / "single_gpu.jsonl"
        self.gpu = int(cfg.get("gpu", 0))
        self.env = {"CUDA_VISIBLE_DEVICES": str(self.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.repos = {k: expand(v) for k, v in cfg["repos"].items()}
        self.k = int(cfg.get("k", 10))
        self.R = int(cfg.get("degree", 64))
        self.threads = int(cfg.get("cpu_threads", 0)) or os.cpu_count()
        self.done = {self._key(r) for r in load_jsonl(self.results) if r.get("ok")}
        self.ginfo = gpu_info(self.gpu)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _key(r):
        return (r["kind"], r["dataset"], r["system"], r.get("variant", ""), r.get("searcher", ""),
                r.get("dtype", ""))

    def _record(self, rec):
        rec.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        rec.setdefault("gpu_name", self.ginfo.get("gpu_name"))
        append_jsonl(self.results, rec)
        if rec.get("ok"):
            self.done.add(self._key(rec))

    def _is_done(self, **key):
        rec = dict(key)
        return not self.force and self._key(rec) in self.done

    def _run(self, cmd, log, monitor=True, cwd=None):
        print(f"  $ {' '.join(map(str, cmd))}", flush=True)
        if monitor:
            with GpuMemMonitor(self.gpu) as m:
                rc, wall, out = run_logged(cmd, log, env=self.env, cwd=cwd)
            return rc, wall, out, m.peak_gb
        rc, wall, out = run_logged(cmd, log, env=self.env, cwd=cwd)
        return rc, wall, out, None

    def _fail(self, base, rc, out, log):
        tail = "\n".join(out.strip().splitlines()[-15:])
        print(f"  !! failed (rc={rc}); see {log}\n{tail}")
        self._record(dict(base, ok=False, rc=rc, log=str(log), error_tail=tail))

    # ------------------------------------------------------------ searching
    def search_graph(self, ds, meta, system, variant, graph):
        sc = self.cfg["search"]
        dtypes = list(sc.get("dtypes", ["native"]))
        if meta["dtype"] != "float32":
            dtypes = [d for d in dtypes if d == "native"]
        for dt in dtypes:
            base = dict(kind="search", dataset=ds, system=system, variant=variant, searcher="cagra", dtype=dt)
            if self._is_done(**base):
                continue
            d = self.data_root / ds
            log = graph.parent / f"search_cagra_{dt}.log"
            cmd = [PY, HERE / "search_graph.py", "--base", d / meta["base"], "--query", d / meta["query"],
                   "--gt", d / meta["gt"], "--graph", graph, "--k", self.k, "--dtype", dt,
                   "--batch-size", sc.get("batch_size", 10000), "--repeats", sc.get("repeats", 5),
                   "--itopk", *sc["itopk"], "--search-width", *sc["search_width"]]
            rc, wall, out, _ = self._run(cmd, log, monitor=False, cwd=HERE)
            if rc != 0:
                self._fail(base, rc, out, log)
                continue
            self._record(dict(base, ok=True, **parse_emitted(out)))

    # --------------------------------------------------------------- builds
    def build_cagra(self, ds, meta):
        c = self.cfg["systems"]["cagra"]
        variant = c.get("build_algo", "nn_descent")
        wd = self.work_root / ds / f"cagra-{variant}"
        graph = wd / "graph.u32bin"
        base = dict(kind="build", dataset=ds, system="cagra", variant=variant)
        if not self._is_done(**base) or not graph.exists():
            wd.mkdir(parents=True, exist_ok=True)
            cmd = [PY, HERE / "build.py", "cagra", "--base", self.data_root / ds / meta["base"], "--out", graph,
                   "--degree", self.R, "--inter-degree", c.get("intermediate_degree", 128), "--algo", variant]
            rc, wall, out, peak = self._run(cmd, wd / "build.log", cwd=HERE)
            if rc != 0:
                return self._fail(base, rc, out, wd / "build.log")
            r = parse_emitted(out)
            self._record(dict(base, ok=True, build_core_s=r["build_core_s"], build_e2e_s=wall,
                              peak_gpu_mem_gb=peak, stages=r["stages"], graph=r["graph"],
                              params=dict(degree=self.R, intermediate_degree=c.get("intermediate_degree", 128))))
        self.search_graph(ds, meta, "cagra", variant, graph)

    def build_jasper(self, ds, meta):
        c = self.cfg["systems"]["jasper"]
        if self.R not in (32, 64):
            print(f"  skip jasper: degree {self.R} not compiled (32/64 only)")
            return
        variant = f"R{self.R}"
        wd = self.work_root / ds / f"jasper-{variant}"
        wd.mkdir(parents=True, exist_ok=True)
        idx = wd / "index.jasper"
        graph = wd / "graph.u32bin"
        bindir = self.repos["jasper"] / "build-eval" / "bin"
        d = self.data_root / ds
        dtype = "float" if meta["dtype"] == "float32" else "uint8"
        base = dict(kind="build", dataset=ds, system="jasper", variant=variant)
        if not self._is_done(**base) or not idx.exists():
            cmd = [bindir / "create_index", "-f", d / meta["base"], "-t", dtype, "-d", "l2", "-n", self.R,
                   "--dim", meta["dim"], "-a", c.get("alpha", 1.2), "-w", c.get("workspace_budget", "10GB"),
                   "-i", idx]
            rc, wall, out, peak = self._run(cmd, wd / "build.log")
            if rc != 0:
                return self._fail(base, rc, out, wd / "build.log")
            core_ms = parse_float(r"timing summary \(\d+ batches, ([\d.]+) ms total\)", out)
            phases = {m[0].strip().replace(" ", "_").replace("+", "and"): float(m[1])
                      for m in re.findall(r"^\s+([a-z +]+?)\s*:\s+([\d.]+) ms", out, re.M)}
            self._record(dict(base, ok=True, build_core_s=core_ms / 1e3 if core_ms else None, build_e2e_s=wall,
                              peak_gpu_mem_gb=peak, stages={f"{k}_ms": v for k, v in phases.items()},
                              params=dict(R=self.R, alpha=c.get("alpha", 1.2), L=64)))
        # native beam search
        sbase = dict(kind="search", dataset=ds, system="jasper", variant=variant, searcher="native", dtype="f16")
        if not self._is_done(**sbase):
            bl = [f"{bw}:{2 * bw}" for bw in c["beam_widths"] if bw >= self.k]
            cmd = [bindir / "run_query", "-i", idx, "-q", d / meta["query"], "-g", d / meta["gt_jasper"],
                   "-t", dtype, "-d", "l2", "-n", self.R, "--dim", meta["dim"], "-k", self.k, "-b", *bl]
            rc, wall, out, _ = self._run(cmd, wd / "search_native.log", monitor=False)
            if rc != 0:
                self._fail(sbase, rc, out, wd / "search_native.log")
            else:
                pts = []
                pat = re.compile(r"beam_width=(\d+) limit=(\d+) duration=([\d.]+)ms throughput=(\d+) QPS\s*\n"
                                 r"\s*Recall@\d+:\s*([\d.]+)")
                for bw, lim, dur, qps, rec in pat.findall(out):
                    pts.append(dict(beam_width=int(bw), limit=int(lim), time_s=float(dur) / 1e3,
                                    qps=float(qps), recall=float(rec)))
                self._record(dict(sbase, ok=bool(pts), points=pts, batch_size=meta["n_queries"], k=self.k,
                                  note="single batch, kernel time (cudaEvent), one warmup"))
        # export graph -> unified CAGRA search
        if not graph.exists() or self.force:
            cmd = [PY, HERE / "build.py", "export-jasper", "--index", idx, "--dim", meta["dim"], "--R", self.R,
                   "--out", graph]
            rc, _, out, _ = self._run(cmd, wd / "export.log", monitor=False, cwd=HERE)
            if rc != 0:
                return self._fail(dict(kind="export", dataset=ds, system="jasper", variant=variant), rc, out,
                                  wd / "export.log")
            gs = parse_emitted(out)
            (wd / "graph_stats.json").write_text(json.dumps(gs))
            self._record(dict(kind="export", dataset=ds, system="jasper", variant=variant, ok=True, **gs))
        self.search_graph(ds, meta, "jasper", variant, graph)

    def build_tagore(self, ds, meta):
        c = self.cfg["systems"]["tagore"]
        if not meta.get("fvecs"):
            print(f"  skip tagore: dim {meta['dim']} unsupported (needs multiple of 32, <= 128)")
            return
        for v in c["variants"]:
            variant = v["name"]
            wd = self.work_root / ds / f"tagore-{variant}"
            graph = wd / "graph.u32bin"
            base = dict(kind="build", dataset=ds, system="tagore", variant=variant)
            if not self._is_done(**base) or not graph.exists():
                wd.mkdir(parents=True, exist_ok=True)
                cmd = [PY, HERE / "build.py", "tagore", "--tagore-build", self.repos["tagore"] / "build-eval",
                       "--fvecs", self.data_root / ds / meta["fvecs"], "--n", meta["n"], "--dim", meta["dim"],
                       "--knn-k", c.get("knn_k", 96), "--iters", c.get("iters", 10), "--degree", self.R,
                       "--m", c.get("m", 64), "--prune", v["prune"], "--alpha", v.get("alpha", 1.0),
                       "--raw-out", wd / "index.tagore", "--out", graph]
                rc, wall, out, peak = self._run(cmd, wd / "build.log", cwd=HERE)
                if rc != 0:
                    self._fail(base, rc, out, wd / "build.log")
                    continue
                r = parse_emitted(out)
                knn = parse_float(r"time of GNN-Descent: ([\d.eE+-]+)s", out)
                prune = parse_float(r"time of pruning: ([\d.eE+-]+)s", out)
                stages = dict(r["stages"], gnn_descent_s=knn, pruning_s=prune)
                self._record(dict(base, ok=True, build_core_s=(knn or 0) + (prune or 0), build_e2e_s=wall,
                                  peak_gpu_mem_gb=peak, stages=stages, graph=r["graph"],
                                  params=dict(knn_k=c.get("knn_k", 96), iters=c.get("iters", 10), m=c.get("m", 64),
                                              degree=self.R, prune=v["prune"], alpha=v.get("alpha", 1.0)),
                                  note="pruning_s includes D2H copy and index file write"))
            self.search_graph(ds, meta, "tagore", variant, graph)

    def build_scalegann(self, ds, meta):
        c = self.cfg["systems"]["scalegann"]
        bindir = self.repos["scalegann"] / "build-eval"
        inter = c.get("intermediate_degree", 128)
        algo = c.get("build_algo", "nn_descent")
        for v in c["variants"]:
            variant = v["name"]
            wd = self.work_root / ds / f"scalegann-{variant}"
            graph = wd / "graph.u32bin"
            base = dict(kind="build", dataset=ds, system="scalegann", variant=variant)
            if not self._is_done(**base) or not graph.exists():
                if wd.exists():
                    shutil.rmtree(wd)
                parts = wd / "parts"
                parts.mkdir(parents=True)
                mem = v.get("memory_gb", "auto")
                if mem == "auto":
                    mem = int(self.ginfo.get("gpu_mem_gb", 16))
                index_name = f"cagra_R{self.R}"
                # 1) partition (CPU k-means + selective duplication)
                cmd = [bindir / "executeDiskPartition", "--data_path", self.data_root / ds / meta["base"],
                       "--base_folder", parts, "-M", mem, "-R", self.R, "-L", inter,
                       "-W", c.get("max_duplication", 2), "-E", c.get("epsilon", 1.2), "-T", self.threads,
                       "-N", v.get("min_partitions", 1)]
                rc, w_part, out, _ = self._run(cmd, wd / "partition.log", monitor=False)
                if rc != 0:
                    self._fail(base, rc, out, wd / "partition.log")
                    continue
                part_s = parse_float(r"Disk partition duration: (\d+) milliseconds", out) / 1e3
                # 2) per-shard CAGRA build (sequential on one GPU)
                cmd = [PY, HERE / "build.py", "scalegann-shards", "--base-folder", parts, "--index-name", index_name,
                       "--base", self.data_root / ds / meta["base"],
                       "--degree", self.R, "--inter-degree", inter, "--algo", algo]
                rc, w_build, out, peak = self._run(cmd, wd / "shards.log", cwd=HERE)
                if rc != 0:
                    self._fail(base, rc, out, wd / "shards.log")
                    continue
                sh = parse_emitted(out)
                # 3) merge
                cmd = [bindir / "executeDiskMerge", "--base_folder", parts, "--index_name", index_name,
                       "-R", self.R, "-B", self.R, "-T", self.threads]
                rc, w_merge, out, _ = self._run(cmd, wd / "merge.log", monitor=False)
                if rc != 0:
                    self._fail(base, rc, out, wd / "merge.log")
                    continue
                merge_s = parse_float(r"overall duration: (\d+) milliseconds", out) / 1e3
                # 4) export
                cmd = [PY, HERE / "build.py", "export-scalegann", "--index", parts / "mergedIndex" / index_name,
                       "--out", graph]
                rc, _, out, _ = self._run(cmd, wd / "export.log", monitor=False, cwd=HERE)
                if rc != 0:
                    self._fail(base, rc, out, wd / "export.log")
                    continue
                g = parse_emitted(out)
                stored = sum(sh["shard_sizes"])
                self._record(dict(base, ok=True,
                                  build_core_s=part_s + sh["shards_build_sum_s"] + merge_s,
                                  build_e2e_s=w_part + w_build + w_merge, peak_gpu_mem_gb=peak,
                                  stages=dict(partition_s=part_s, shard_build_s=sh["shards_build_sum_s"],
                                              shard_wall_s=sh["shards_wall_sum_s"], merge_s=merge_s),
                                  n_shards=sh["n_shards"], shard_sizes=sh["shard_sizes"],
                                  duplication=stored / meta["n"], graph=g["graph"],
                                  params=dict(memory_gb=mem, min_partitions=v.get("min_partitions", 1),
                                              R=self.R, L=inter, W=c.get("max_duplication", 2),
                                              E=c.get("epsilon", 1.2), threads=self.threads)))
                if not c.get("keep_partitions", False):
                    shutil.rmtree(parts, ignore_errors=True)
            self.search_graph(ds, meta, "scalegann", variant, graph)

    # ------------------------------------------------------------------ run
    def environment(self):
        env = dict(self.ginfo, host=platform.node(), cpu_count=os.cpu_count(), python=platform.python_version())
        try:
            import cuvs
            env["cuvs_version"] = cuvs.__version__
        except Exception:
            pass
        try:
            env["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                    if l.startswith("model name"))
        except Exception:
            pass
        for name, path in self.repos.items():
            try:
                env[f"{name}_commit"] = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
            except Exception:
                pass
        return env

    def run(self, datasets, systems):
        env = self.environment()
        print(json.dumps(env, indent=2))
        (self.work_root / "results").mkdir(parents=True, exist_ok=True)
        (self.work_root / "results" / "environment.json").write_text(json.dumps(env, indent=2))
        for ds in datasets:
            meta_p = self.data_root / ds / "meta.json"
            if not meta_p.exists():
                print(f"!! {ds}: {meta_p} missing; run prepare_data.py first")
                continue
            meta = json.loads(meta_p.read_text())
            for s in systems:
                if not self.cfg["systems"].get(s, {}).get("enabled", True):
                    continue
                print(f"\n=== {ds} / {s} ===", flush=True)
                getattr(self, f"build_{s}")(ds, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=str(HERE / "configs" / "single_gpu.yaml"))
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS)
    ap.add_argument("--force", action="store_true", help="redo steps that already have results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    Runner(cfg, force=a.force).run(a.datasets or cfg["datasets"], a.systems or SYSTEMS)


if __name__ == "__main__":
    main()
