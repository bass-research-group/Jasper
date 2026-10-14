"""Shared helpers for the GPU ANNS evaluation harness.

File formats
------------
* ``*.fbin`` / ``*.u8bin`` / ``*.i8bin`` / ``*.ibin`` / ``*.u32bin``:
  big-ann-benchmarks layout, ``[uint32 n][uint32 dim]`` followed by row-major data.
  This is the canonical format for base/query/groundtruth and for the
  fixed-degree graphs every builder exports (``graph.u32bin``).
* ``*.fvecs``: per-row ``[int32 dim][dim x float32]`` (Tagore input).
* ``gt.jasper``: ``[u32 nq][u32 k][u32 ids nq*k][f32 dists nq*k]`` (Jasper run_query).
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SUFFIX_DTYPE = {
    ".fbin": np.float32,
    ".f16bin": np.float16,
    ".u8bin": np.uint8,
    ".i8bin": np.int8,
    ".ibin": np.int32,
    ".u32bin": np.uint32,
}


def bin_dtype(path):
    return SUFFIX_DTYPE[Path(path).suffix]


def read_bin(path, mmap=False, max_rows=None):
    path = str(path)
    dtype = bin_dtype(path)
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    n, d = int(n), int(d)
    if max_rows is not None:
        n = min(n, max_rows)
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))
    return np.fromfile(path, dtype=dtype, count=n * d, offset=8).reshape(n, d)


def bin_shape(path):
    n, d = np.fromfile(str(path), dtype=np.uint32, count=2)
    return int(n), int(d)


def write_bin(path, arr):
    arr = np.ascontiguousarray(arr)
    assert arr.ndim == 2
    expected = SUFFIX_DTYPE.get(Path(path).suffix)
    if expected is not None:
        assert arr.dtype == expected, f"{path}: dtype {arr.dtype} != {np.dtype(expected)}"
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.array(arr.shape, dtype=np.uint32).tofile(f)
        arr.tofile(f)
    os.replace(tmp, path)


def write_fvecs(path, arr, chunk=1 << 20):
    arr = np.asarray(arr)
    n, d = arr.shape
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        for s in range(0, n, chunk):
            blk = np.asarray(arr[s:s + chunk], dtype=np.float32)
            out = np.empty((blk.shape[0], d + 1), dtype=np.float32)
            out[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
            out[:, 1:] = blk
            out.tofile(f)
    os.replace(tmp, path)


def read_vecs(path, dtype):
    """Read .fvecs / .ivecs / .bvecs."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    d = int(raw[:4].view(np.int32)[0])
    item = np.dtype(dtype).itemsize
    row = 4 + d * item
    rows = raw.reshape(-1, row)[:, 4:]
    return np.ascontiguousarray(rows).view(dtype).reshape(-1, d)


def write_jasper_gt(path, ids, dists):
    nq, k = ids.shape
    with open(path, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        np.ascontiguousarray(ids, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(dists, dtype=np.float32).tofile(f)


def recall_at_k(found, gt, k):
    found = np.asarray(found)[:, :k]
    gt = np.asarray(gt)[:, :k]
    hits = 0
    for f, g in zip(found, gt):
        hits += len(np.intersect1d(f, g, assume_unique=False))
    return hits / (gt.shape[0] * k)


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------

def pad_rows(edges, counts, n_ids):
    """Turn a variable-degree adjacency (edges[n, R], counts[n]) into a fixed
    degree uint32 graph usable by cuvs.cagra.from_graph.

    Slots past ``counts`` and invalid ids are filled by cycling through the
    row's valid neighbours (duplicates are harmless for CAGRA search: visited
    nodes are filtered by its hash table). Rows with no valid neighbour point
    at themselves.
    """
    edges = np.array(edges, dtype=np.uint32, copy=True)
    n, R = edges.shape
    counts = np.minimum(np.asarray(counts, dtype=np.int64), R)
    col = np.arange(R)[None, :]
    valid = (col < counts[:, None]) & (edges < n_ids)
    # Compact valid entries to the front of each row (stable).
    order = np.argsort(~valid, axis=1, kind="stable")
    edges = np.take_along_axis(edges, order, axis=1)
    nvalid = valid.sum(axis=1)
    empty = nvalid == 0
    nv = np.maximum(nvalid, 1)
    src = col % nv[:, None]
    edges = np.take_along_axis(edges, src, axis=1)
    edges[empty] = np.arange(n, dtype=np.uint32)[empty, None]
    return edges, nvalid


def graph_stats(graph, nvalid=None):
    g = np.asarray(graph)
    out = {"n": int(g.shape[0]), "degree": int(g.shape[1])}
    if nvalid is not None:
        nvalid = np.asarray(nvalid)
        out.update(avg_out_degree=float(nvalid.mean()), min_out_degree=int(nvalid.min()),
                   max_out_degree=int(nvalid.max()))
    return out


# --------------------------------------------------------------------------
# Process / timing helpers
# --------------------------------------------------------------------------

class GpuMemMonitor:
    """Polls device memory usage of one physical GPU (NVML index) and records
    the peak above the pre-launch baseline. Other tenants on the GPU will
    pollute this number; it is a coarse indicator only."""

    def __init__(self, nvml_index, interval=0.1):
        self.idx = nvml_index
        self.interval = interval
        self.peak = 0
        self.base = 0
        self._stop = threading.Event()
        self._ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            self._nvml = pynvml
            self._ok = True
        except Exception:
            self._ok = False

    def _used(self):
        return self._nvml.nvmlDeviceGetMemoryInfo(self._h).used

    def __enter__(self):
        if self._ok:
            self.base = self._used()
            self.peak = self.base
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._used())
            except Exception:
                pass
            time.sleep(self.interval)

    def __exit__(self, *exc):
        if self._ok:
            self._stop.set()
            self._t.join()

    @property
    def peak_gb(self):
        if not self._ok:
            return None
        return round((self.peak - self.base) / 2**30, 3)


def gpu_info(nvml_index):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).total
        drv = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(drv, bytes):
            drv = drv.decode()
        return {"gpu_name": name, "gpu_mem_gb": round(mem / 2**30, 1), "driver": drv}
    except Exception as e:  # pragma: no cover
        return {"gpu_name": "unknown", "error": str(e)}


def run_logged(cmd, log_path, env=None, cwd=None, timeout=None):
    """Run ``cmd`` (list), tee stdout+stderr to ``log_path``, return
    (returncode, wall_seconds, output_text)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    t0 = time.perf_counter()
    out_chunks = []
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=full_env, cwd=cwd,
                             text=True, bufsize=1)
        try:
            for line in p.stdout:
                log.write(line)
                out_chunks.append(line)
                sys.stdout.write("    | " + line)
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            out_chunks.append("\n[harness] TIMEOUT\n")
    return p.returncode, time.perf_counter() - t0, "".join(out_chunks)


def parse_float(pattern, text, default=None, last=True):
    m = re.findall(pattern, text)
    if not m:
        return default
    v = m[-1] if last else m[0]
    if isinstance(v, tuple):
        v = v[0]
    return float(v)


def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def emit_result(obj):
    """Builders print one machine-readable line that the orchestrator parses."""
    print("@@RESULT " + json.dumps(obj), flush=True)


def parse_emitted(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("@@RESULT "):
            out.update(json.loads(line[len("@@RESULT "):]))
    return out
