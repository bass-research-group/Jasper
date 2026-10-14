"""Readers/writers for the graph formats of each system.

Every reader returns ``(graph uint32[n, R], nvalid int[n])`` where rows are
padded to a fixed degree (see ``common.pad_rows``) so the graph can be loaded
with ``cuvs.neighbors.cagra.from_graph`` and searched by the same GPU kernel.
"""

import ast
import os
import tempfile

import numpy as np

from common import pad_rows

# --------------------------------------------------------------------------
# numpy-stream (RAFT / cuVS serialization) helpers
# --------------------------------------------------------------------------

NPY_MAGIC = b"\x93NUMPY"
_PREAMBLE = 128  # fixed so every shard file has an identical header size


def _npy_write(f, arr):
    arr = np.asarray(arr)
    descr = np.lib.format.dtype_to_descr(arr.dtype)
    hdr = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (descr, tuple(arr.shape))
    body_len = _PREAMBLE - len(NPY_MAGIC) - 2 - 2
    hdr = hdr.ljust(body_len - 1) + "\n"
    assert len(hdr) == body_len, "npy header too long"
    f.write(NPY_MAGIC + b"\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode("latin1"))
    f.write(np.ascontiguousarray(arr).tobytes())


def _npy_read(f):
    magic = f.read(6)
    if magic != NPY_MAGIC:
        raise ValueError(f"bad npy magic {magic!r} at {f.tell() - 6}")
    major = f.read(2)[0]
    hlen_bytes = 2 if major == 1 else 4
    hlen = int.from_bytes(f.read(hlen_bytes), "little")
    hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
    dt = np.dtype(hdr["descr"])
    shape = tuple(hdr["shape"])
    count = int(np.prod(shape)) if shape else 1
    data = np.frombuffer(f.read(count * dt.itemsize), dtype=dt)
    return data.reshape(shape) if shape else data[0]


def write_raft_cagra_graph(path, graph, dim=0):
    """Write a graph-only CAGRA index in the RAFT 24.10 layout that ScaleGANN's
    ``executeDiskMerge`` / ``readIndex`` parse:
    dtype[4] | npy int32 version | npy u32 rows | npy u32 dim | npy u32 deg |
    npy u16 metric | npy u32 graph[rows,deg] | npy bool has_dataset
    """
    graph = np.ascontiguousarray(graph, dtype=np.uint32)
    rows, deg = graph.shape
    with open(path, "wb") as f:
        f.write(b"<f4\x00")
        _npy_write(f, np.array(4, dtype=np.int32))
        _npy_write(f, np.array(rows, dtype=np.uint32))
        _npy_write(f, np.array(dim, dtype=np.uint32))
        _npy_write(f, np.array(deg, dtype=np.uint32))
        _npy_write(f, np.array(0, dtype=np.uint16))  # L2Expanded
        _npy_write(f, graph)
        _npy_write(f, np.array(False, dtype=np.bool_))


def read_npy_stream_graph(path):
    """Read the first 2-D uint32 array in a RAFT/cuVS serialized CAGRA file.
    Works for ScaleGANN merged indexes and (format permitting) cuVS saves."""
    with open(path, "rb") as f:
        f.read(4)  # dtype string
        while True:
            a = _npy_read(f)
            if isinstance(a, np.ndarray) and a.ndim == 2 and a.dtype == np.uint32:
                return np.array(a)


# --------------------------------------------------------------------------
# cuVS CAGRA
# --------------------------------------------------------------------------

def cagra_index_graph(index):
    """uint32[n, deg] host copy of a built cuvs CAGRA index's graph."""
    try:
        import cupy as cp
        return cp.asnumpy(cp.asarray(index.graph)).astype(np.uint32, copy=False)
    except Exception:
        pass
    # Fallback for cuvs versions without Index.graph: save graph only and parse.
    from cuvs.neighbors import cagra
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "idx")
        cagra.save(p, index, include_dataset=False)
        return read_npy_stream_graph(p)


# --------------------------------------------------------------------------
# Jasper
# --------------------------------------------------------------------------

def read_jasper_graph(path, dim, R):
    """Jasper index file (include/jasper/index/graph.cuh save_graph_to_file):
    [u64 file_size][u64 n][u64 medoid][u64 bytes_per_node] then n records of
    fp16 vec[dim] | u8 edge_count | u32 edges[R] | bf16 dist[R] | u32 slot_to_id
    """
    hdr = np.fromfile(path, dtype=np.uint64, count=4)
    n, medoid, bpn = int(hdr[1]), int(hdr[2]), int(hdr[3])
    rec = np.dtype([("vec", "<f2", (dim,)), ("cnt", "u1"), ("edges", "<u4", (R,)),
                    ("dist", "<u2", (R,)), ("id", "<u4")])
    if rec.itemsize != bpn:
        raise ValueError(f"Jasper record size {rec.itemsize} != bytes_per_node {bpn} "
                         f"(dim={dim}, R={R}); legacy layout not supported")
    m = np.memmap(path, dtype=rec, mode="r", offset=32, shape=(n,))
    slot_to_id = np.array(m["id"])
    edges = np.array(m["edges"])
    cnt = np.array(m["cnt"]).astype(np.int64)
    # Edges reference slots; translate to stable ids and order rows by id.
    valid_slot = edges < n
    edges = np.where(valid_slot, slot_to_id[np.minimum(edges, n - 1)], np.uint32(0xFFFFFFFF))
    order = np.argsort(slot_to_id, kind="stable")
    if not np.array_equal(slot_to_id[order], np.arange(n, dtype=np.uint32)):
        raise ValueError("Jasper slot_to_id is not a permutation of 0..n-1")
    g, nvalid = pad_rows(edges[order], cnt[order], n)
    return g, nvalid, int(slot_to_id[medoid])


# --------------------------------------------------------------------------
# Tagore
# --------------------------------------------------------------------------

def read_tagore_cagra(path, n):
    """StoreCAGRA: n records of [u32 d][u32 ids[d]] (fixed d, no header)."""
    raw = np.fromfile(path, dtype=np.uint32)
    d = int(raw[0])
    assert raw.size == n * (d + 1), f"unexpected Tagore CAGRA file size ({raw.size} words, n={n}, d={d})"
    rows = raw.reshape(n, d + 1)
    cnt = rows[:, 0].astype(np.int64)
    return pad_rows(rows[:, 1:], cnt, n)


def read_diskann_mem_graph(path, n=None):
    """DiskANN in-memory graph (Tagore StoreVamana):
    [u64 file_size][u32 max_degree][u32 ep][u64 frozen] then per node [u32 deg][u32 ids[deg]]."""
    raw = np.fromfile(path, dtype=np.uint32)
    max_deg, ep = int(raw[2]), int(raw[3])
    pos = 6
    total = raw.size
    rows, cnts = [], []
    while pos < total:
        k = int(raw[pos])
        rows.append(raw[pos + 1:pos + 1 + k])
        cnts.append(k)
        pos += 1 + k
    if n is not None:
        assert len(rows) == n, f"graph has {len(rows)} nodes, expected {n}"
    R = max(max_deg, max(cnts))
    edges = np.full((len(rows), R), 0xFFFFFFFF, dtype=np.uint32)
    for i, r in enumerate(rows):
        edges[i, :len(r)] = r
    g, nvalid = pad_rows(edges, np.array(cnts), len(rows))
    return g, nvalid, ep
