import ctypes
import os

import numpy as np

from ..base.module import BaseANN

# The bridge library is baked into the image outside any bind-mounted path
# (the VIBE repository and the home directory are mounted by Singularity, so
# a library placed there would be shadowed at runtime). HANN_BRIDGE_LIB
# overrides the path for local testing.
DEFAULT_LIB_PATH = "/opt/hann/libhann_bridge.so"


def _load_bridge():
    lib_path = os.environ.get("HANN_BRIDGE_LIB", DEFAULT_LIB_PATH)
    lib = ctypes.CDLL(lib_path)

    lib.hann_hnsw_new.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p]
    lib.hann_hnsw_new.restype = ctypes.c_int64

    lib.hann_pqivf_new.argtypes = [ctypes.c_int64] * 6
    lib.hann_pqivf_new.restype = ctypes.c_int64

    lib.hann_rpt_new.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_char_p,
    ]
    lib.hann_rpt_new.restype = ctypes.c_int64

    lib.hann_add_batch.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.hann_add_batch.restype = ctypes.c_int64

    lib.hann_train.argtypes = [ctypes.c_int64]
    lib.hann_train.restype = ctypes.c_int64

    lib.hann_hnsw_set_ef.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.hann_hnsw_set_ef.restype = ctypes.c_int64

    lib.hann_search.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.hann_search.restype = ctypes.c_int64

    lib.hann_free.argtypes = [ctypes.c_int64]
    lib.hann_free.restype = None

    return lib


class _HannBase(BaseANN):
    """Shared handle management, insertion, and search for the Hann indexes."""

    def __init__(self):
        self.lib = _load_bridge()
        self.handle = None
        self.dim = None

    def _new_handle(self, dim):
        raise NotImplementedError

    def _prepare(self, X):
        """Hook for per-index preprocessing of the input matrix."""
        return X

    def fit(self, X):
        X = np.ascontiguousarray(self._prepare(X), dtype=np.float32)
        n, dim = X.shape
        self.dim = dim
        self.handle = self._new_handle(dim)
        if self.handle < 0:
            raise RuntimeError("hann: failed to create the index")
        added = self.lib.hann_add_batch(
            self.handle, X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, dim
        )
        if added != n:
            raise RuntimeError("hann: added %d of %d vectors" % (added, n))

    def query(self, v, n):
        q = np.ascontiguousarray(self._prepare(v.reshape(1, -1))[0], dtype=np.float32)
        out = np.empty(n, dtype=np.int32)
        count = self.lib.hann_search(
            self.handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.dim,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if count < 0:
            raise RuntimeError("hann: search failed")
        return out[:count]

    def freeIndex(self):
        if self.handle is not None:
            self.lib.hann_free(self.handle)
            self.handle = None

    def done(self):
        self.freeIndex()


class HannHNSW(_HannBase):
    def __init__(self, metric, M, efConstruction):
        # "normalized" vectors are pre-normalized, so cosine ordering equals
        # inner-product ordering. Plain "ip" has no counterpart in Hann.
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"HannHNSW does not support metric {metric}")
        super().__init__()
        self.metric = {"euclidean": "euclidean", "cosine": "cosine", "normalized": "cosine"}[metric]
        self.M = M
        self.efConstruction = efConstruction
        self.ef_query = None

    def _new_handle(self, dim):
        return self.lib.hann_hnsw_new(
            dim, self.M, self.efConstruction, self.metric.encode("ascii")
        )

    def _push_ef(self, ef):
        if self.lib.hann_hnsw_set_ef(self.handle, ef) != 0:
            raise RuntimeError("hann: failed to set ef to %d" % ef)
        self._ef_current = ef

    def set_query_arguments(self, ef):
        self._push_ef(ef)
        self.ef_query = ef

    def query(self, v, n):
        # Hann searches with the configured ef even when it is below k, and
        # then completes the result with an exact scan. hnswlib instead
        # searches with max(ef, k). Clamp the same way, so low-ef points
        # measure the graph rather than the scan.
        if self._ef_current < n:
            self._push_ef(n)
        return super().query(v, n)

    def __str__(self):
        return "HannHNSW(M=%d, efConstruction=%d, efQuery=%d)" % (
            self.M,
            self.efConstruction,
            self.ef_query,
        )


class HannPQIVF(_HannBase):
    def __init__(self, metric, coarseK, candidateClusters):
        # The PQIVF index is Euclidean only. Euclidean ordering on unit
        # vectors equals cosine ordering, so cosine data is normalized here
        # and "normalized" data is already unit length.
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"HannPQIVF does not support metric {metric}")
        super().__init__()
        self.normalize = metric == "cosine"
        self.coarseK = coarseK
        self.candidateClusters = candidateClusters

    def _prepare(self, X):
        if not self.normalize:
            return X
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _new_handle(self, dim):
        # Zeros keep the library defaults for numSubquantizers, pqK, and
        # kMeansIters.
        return self.lib.hann_pqivf_new(dim, self.coarseK, 0, 0, 0, self.candidateClusters)

    def fit(self, X):
        super().fit(X)
        if self.lib.hann_train(self.handle) != 0:
            raise RuntimeError("hann: training failed")

    def __str__(self):
        return "HannPQIVF(coarseK=%d, candidateClusters=%d)" % (
            self.coarseK,
            self.candidateClusters,
        )


class HannRPT(_HannBase):
    def __init__(self, metric, leafCapacity, probeMargin):
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"HannRPT does not support metric {metric}")
        super().__init__()
        self.metric = {"euclidean": "euclidean", "cosine": "cosine", "normalized": "cosine"}[metric]
        self.leafCapacity = leafCapacity
        self.probeMargin = probeMargin

    def _new_handle(self, dim):
        # A zero keeps the library default for candidateProjections.
        return self.lib.hann_rpt_new(
            dim, self.leafCapacity, 0, self.probeMargin, self.metric.encode("ascii")
        )

    def __str__(self):
        return "HannRPT(leafCapacity=%d, probeMargin=%g)" % (
            self.leafCapacity,
            self.probeMargin,
        )
