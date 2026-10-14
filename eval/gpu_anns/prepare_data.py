#!/usr/bin/env python3
"""Download / convert datasets and compute exact ground truth on the GPU.

For every dataset this produces, under ``<data_root>/<name>/``::

    base.{fbin,u8bin}     big-ann layout (input to Jasper, CAGRA, ScaleGANN)
    query.{fbin,u8bin}
    gt.ibin / gt.dist.fbin  exact top-100 (int32 ids / float32 squared L2)
    gt.jasper             same ground truth in Jasper's run_query format
    base.fvecs            float32 fvecs (Tagore input; only when Tagore can
                          handle the dim: multiple of 32 and <= 128)
    meta.json

Examples::

    python prepare_data.py --data-root /data/anns sift-1M deep-10M bigann-10M
    python prepare_data.py --data-root /data/anns deep-10M --base-src /local/deep1b/base.1B.fbin
"""

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from common import read_bin, read_vecs, write_bin, write_fvecs, write_jasper_gt

BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann"
DEEP = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"
TEXMEX_SIFT = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

# name -> spec. "bigann" style specs take the first n rows of a 1B file via an
# HTTP range request, so a 10M subset downloads 1.3 GB / 3.8 GB, not the full file.
DATASETS = {
    "sift-1M": dict(kind="texmex", dtype="float32", dim=128),
    "deep-1M": dict(kind="range", base=f"{DEEP}/base.1B.fbin", query=f"{DEEP}/query.public.10K.fbin",
                    n=1_000_000, dtype="float32", dim=96),
    "deep-10M": dict(kind="range", base=f"{DEEP}/base.1B.fbin", query=f"{DEEP}/query.public.10K.fbin",
                     n=10_000_000, dtype="float32", dim=96),
    "bigann-1M": dict(kind="range", base=f"{BIGANN}/base.1B.u8bin", query=f"{BIGANN}/query.public.10K.u8bin",
                      n=1_000_000, dtype="uint8", dim=128),
    "bigann-10M": dict(kind="range", base=f"{BIGANN}/base.1B.u8bin", query=f"{BIGANN}/query.public.10K.u8bin",
                       n=10_000_000, dtype="uint8", dim=128),
    "bigann-100M": dict(kind="range", base=f"{BIGANN}/base.1B.u8bin", query=f"{BIGANN}/query.public.10K.u8bin",
                        n=100_000_000, dtype="uint8", dim=128),
}

SUFFIX = {"float32": ".fbin", "uint8": ".u8bin"}
GT_K = 100


def download(url, dst, byte_range=None):
    dst = Path(dst)
    if dst.exists():
        return
    print(f"  downloading {url}" + (f" bytes={byte_range[0]}-{byte_range[1]}" if byte_range else ""))
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 24)
    tmp.rename(dst)


def fetch_range_subset(url, n, dim, dtype, dst):
    """First n rows of a big-ann 1B file -> dst (with a corrected header)."""
    dst = Path(dst)
    if dst.exists():
        return
    item = np.dtype(dtype).itemsize
    raw = dst.with_suffix(".raw")
    download(url, raw, byte_range=(8, 8 + n * dim * item - 1))
    assert raw.stat().st_size == n * dim * item, f"short download for {url}"
    with open(dst.with_suffix(".tmp"), "wb") as f, open(raw, "rb") as r:
        np.array([n, dim], dtype=np.uint32).tofile(f)
        shutil.copyfileobj(r, f, length=1 << 24)
    dst.with_suffix(".tmp").rename(dst)
    raw.unlink()


def subset_local(src, n, dst):
    x = read_bin(src, mmap=True, max_rows=n)
    write_bin(dst, np.asarray(x))


def exact_gt(base_path, queries, k=GT_K, chunk_rows=2_000_000):
    """Exact squared-L2 kNN on GPU, chunked over the base set."""
    import cupy as cp
    from cuvs.neighbors import brute_force

    base = read_bin(base_path, mmap=True)
    n = base.shape[0]
    q = cp.asarray(queries, dtype=cp.float32)
    best_d = best_i = None
    for s in range(0, n, chunk_rows):
        blk = cp.asarray(np.asarray(base[s:s + chunk_rows]), dtype=cp.float32)
        idx = brute_force.build(blk, metric="sqeuclidean")
        d, i = brute_force.search(idx, q, min(k, blk.shape[0]))
        d = cp.asarray(d)
        i = cp.asarray(i).astype(cp.int64) + s
        if best_d is None:
            best_d, best_i = d, i
        else:
            cd = cp.concatenate([best_d, d], axis=1)
            ci = cp.concatenate([best_i, i], axis=1)
            o = cp.argsort(cd, axis=1)[:, :k]
            best_d = cp.take_along_axis(cd, o, axis=1)
            best_i = cp.take_along_axis(ci, o, axis=1)
        del idx, blk
        print(f"    gt: {min(s + chunk_rows, n)}/{n}", flush=True)
    return cp.asnumpy(best_i).astype(np.int32), cp.asnumpy(best_d).astype(np.float32)


def prepare(name, root, base_src=None, query_src=None, skip_gt=False):
    spec = DATASETS[name]
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    sfx = SUFFIX[spec["dtype"]]
    base_p, query_p = out / f"base{sfx}", out / f"query{sfx}"
    print(f"== {name} -> {out}")

    if spec["kind"] == "texmex":
        if not base_p.exists():
            tgz = out / "sift.tar.gz"
            download(TEXMEX_SIFT, tgz)
            with tarfile.open(tgz) as t:
                t.extractall(out)
            write_bin(base_p, read_vecs(out / "sift/sift_base.fvecs", np.float32))
            write_bin(query_p, read_vecs(out / "sift/sift_query.fvecs", np.float32))
            shutil.rmtree(out / "sift")
            tgz.unlink()
    else:
        if base_src:
            subset_local(base_src, spec["n"], base_p)
        else:
            fetch_range_subset(spec["base"], spec["n"], spec["dim"], spec["dtype"], base_p)
        if query_src:
            shutil.copy(query_src, query_p)
        else:
            download(spec["query"], query_p)

    n, dim = read_bin(base_p, mmap=True).shape
    queries = read_bin(query_p)
    assert queries.shape[1] == dim

    gt_p = out / "gt.ibin"
    if not gt_p.exists() and not skip_gt:
        print("  computing exact ground truth (GPU)")
        ids, dists = exact_gt(base_p, queries)
        write_bin(out / "gt.dist.fbin", dists)
        write_bin(gt_p, ids)
    if gt_p.exists() and not (out / "gt.jasper").exists():
        ids = read_bin(gt_p)
        dists = read_bin(out / "gt.dist.fbin")
        write_jasper_gt(out / "gt.jasper", ids.astype(np.uint32), dists)

    tagore_ok = dim % 32 == 0 and dim <= 128
    if tagore_ok and not (out / "base.fvecs").exists():
        print("  writing base.fvecs for Tagore")
        write_fvecs(out / "base.fvecs", read_bin(base_p, mmap=True))

    meta = dict(name=name, n=int(n), dim=int(dim), dtype=spec["dtype"], n_queries=int(queries.shape[0]),
                base=base_p.name, query=query_p.name, gt="gt.ibin", gt_jasper="gt.jasper",
                fvecs="base.fvecs" if tagore_ok else None, gt_k=GT_K)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done: n={n} dim={dim} dtype={spec['dtype']} nq={queries.shape[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--base-src", help="local big-ann base file to subset instead of downloading")
    ap.add_argument("--query-src", help="local query file instead of downloading")
    ap.add_argument("--skip-gt", action="store_true")
    a = ap.parse_args()
    for d in a.datasets:
        prepare(d, a.data_root, a.base_src, a.query_src, a.skip_gt)


if __name__ == "__main__":
    main()
