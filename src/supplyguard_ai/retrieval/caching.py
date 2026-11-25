"""
Query-embedding cache.

Found while designing the benchmark: mode="graph" runs one search per related
supplier and per contract edge - each one embedded the SAME query text again
(5-10 identical embedding API calls per question).  Hybrid + pgvector on the
same question repeat it too.  Caching `embed_query` by text removes that cost
and latency; document embedding (ingestion) is passed through unchanged.

Bounded LRU (2,048 texts * 1,536 floats ~ 25 MB worst case), thread-safe
(Chainlit serves users from several threads).  `prime()` lets the benchmark
load vectors saved on disk, so a re-run needs no embedding API at all.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from langchain_core.embeddings import Embeddings


class QueryCachingEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, maxsize: int = 2048) -> None:
        self.inner = inner
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def prime(self, text: str, vector: list[float]) -> None:
        with self._lock:
            self._put(text, list(vector))

    def _put(self, text: str, vector: list[float]) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)  # evict least recently used

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if text in self._cache:
                self.hits += 1
                self._cache.move_to_end(text)
                return list(self._cache[text])
            self.misses += 1
        vector = self.inner.embed_query(text)  # outside the lock: network call
        with self._lock:
            self._put(text, list(vector))
        return list(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)
