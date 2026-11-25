"""
Retrieval service: one search API over several backends (Week 3 day 12 -
Week 4 day 13: pgvector, ChromaDB, hybrid retrieval).

MODES
-----
  "pgvector"  baseline semantic search (cosine distance on vec.documents)
  "chroma"    the same embeddings in ChromaDB (alternative index)
  "keyword"   Postgres full-text search, any-word match ranked by ts_rank
  "hybrid"    Chroma (semantic) + keyword, fused with Reciprocal Rank Fusion
  "graph"     hybrid + Neo4j-guided expansion (GraphRAG): related suppliers and
              the contracts cited by graph edges, all fused with RRF

WHY HYBRID = SEMANTIC + KEYWORD
-------------------------------
Embeddings are good at meaning ("supplier can't deliver" ~ "delay"), weak
at exact identifiers: "PO-2026-9002", "ISO 9001", "LOT-KAT-0932" look alike
to an embedding.  Full-text search is the opposite.  Fusing both covers both
failure modes.  (pgvector + Chroma would be a weak pair: same vectors ->
almost the same ranking.)  The Neo4j graph joins the hybrid path later.

RECIPROCAL RANK FUSION (RRF)
----------------------------
    score(doc) = Σ_backends 1 / (60 + rank_in_that_backend)
Uses ranks, not raw scores, so a cosine distance and a ts_rank never need
to be put on the same scale.  k=60 is the constant from the original paper
(Cormack et al., 2009): it stops a single top-1 hit from dominating.

SECURITY: SCOPES ARE ENFORCED TWICE
-----------------------------------
1. inside each backend query (WHERE access_scope = ANY(scopes) / Chroma where),
   so forbidden chunks never even leave the database;
2. again in Python on every hit (`_allowed`) - defence in depth in case a
   backend filter is ever misconfigured.
The scopes come from the authenticated `User`, never from the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_core.documents import Document
from pydantic import BaseModel

from supplyguard_ai.domain.models import User

Mode = Literal["pgvector", "chroma", "keyword", "hybrid", "graph"]
RRF_K = 60
# Identifiers like HC-026, PO-2026-9001, CERT-S015-21029, MC-17, LOT-KAT-0932
_ID_RE = re.compile(r"\b(?:[A-Z]{2,5}-)+[A-Z]?\d[A-Za-z0-9-]*\b", re.IGNORECASE)


class SearchHit(BaseModel):
    chunk_id: str
    source_id: str
    doc_type: str
    title: str
    supplier_id: str
    version: int
    effective_date: str
    access_scope: str
    superseded_by: str
    contains_instructions: bool
    text: str
    score: float
    backends: list[str]


class Backend(Protocol):
    """Every backend returns Documents RANKED best-first, already scope-filtered."""

    name: str

    def search(self, query: str, k: int, scopes: list[str], where: dict[str, str]) -> list[Document]: ...


# --------------------------------------------------------------------------- backends
@dataclass
class PgVectorBackend:
    store: object  # PGVectorStore
    name: str = "pgvector"

    def search(self, query, k, scopes, where):
        clauses = [{"access_scope": {"$in": scopes}}] + [{key: {"$eq": val}} for key, val in where.items()]
        return self.store.similarity_search(query, k=k, filter={"$and": clauses} if len(clauses) > 1 else clauses[0])


@dataclass
class ChromaBackend:
    store: object  # langchain_chroma.Chroma
    name: str = "chroma"

    def search(self, query, k, scopes, where):
        clauses = [{"access_scope": {"$in": scopes}}] + [{key: val} for key, val in where.items()]
        return self.store.similarity_search(query, k=k, filter={"$and": clauses} if len(clauses) > 1 else clauses[0])


@dataclass
class PgKeywordBackend:
    """Full-text search on the SAME table as pgvector (content column).

    ANY-WORD matching, ranked: the question is parsed with plainto_tsquery
    (never raises on odd punctuation - safe for LLM-written queries) and its
    AND operators are turned into OR; ts_rank then puts documents matching
    more / rarer words first.
    Found by the benchmark: with the default ALL-words matching, a natural
    question ("Is the certificate of X STILL valid?") matched nothing as soon
    as one word ("still") was absent - keyword recall was 0.0 on full
    sentences and only worked for bare ids.
    """

    pool: object
    name: str = "keyword"
    table: str = "documents"  # in schema vec; a code constant, never user input

    _ALLOWED_WHERE = {"doc_type", "supplier_id", "source_id"}

    def search(self, query, k, scopes, where):
        conds, params = ["access_scope = ANY(%(scopes)s)", "to_tsvector('english', content) @@ q.tsq"], \
            {"scopes": scopes, "q": query}
        for key, val in where.items():
            if key not in self._ALLOWED_WHERE:  # column names can't be parameters -> whitelist
                raise ValueError(f"unsupported filter {key}")
            conds.append(f"{key} = %({key})s")
            params[key] = val
        sql = (
            # 'iso' & '9001' & 'valid'  ->  'iso' | '9001' | 'valid'
            "WITH q AS (SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS tsq) "
            "SELECT id, content, source_id, doc_type, title, supplier_id, version, effective_date, access_scope, "
            "superseded_by, chunk_no, contains_instructions, "
            "ts_rank(to_tsvector('english', content), q.tsq) AS rank "
            f"FROM vec.{self.table}, q WHERE {' AND '.join(conds)} ORDER BY rank DESC, id LIMIT %(k)s"
        )
        params["k"] = k
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Document(id=r["id"], page_content=r["content"],
                         metadata={key: r[key] for key in r if key not in ("id", "content", "rank")}) for r in rows]

    def exact(self, ids: list[str], k, scopes, where):
        """Chunks that contain one of `ids` as a whole word (case-insensitive).

        Postgres' text parser splits "PO-2026-9001" into 'po', '-2026', '-9001'
        (negative numbers!), so full-text search can't match an identifier as a
        unit; a word-bounded regex on the raw text can.  `ids` come from
        _ID_RE (letters, digits, hyphens only) and are passed as a parameter.
        """
        conds, params = ["access_scope = ANY(%(scopes)s)", "content ~* %(pattern)s"], \
            {"scopes": scopes, "pattern": r"\m(" + "|".join(ids) + r")\M", "k": k}
        for key, val in where.items():
            if key not in self._ALLOWED_WHERE:
                raise ValueError(f"unsupported filter {key}")
            conds.append(f"{key} = %({key})s")
            params[key] = val
        sql = ("SELECT id, content, source_id, doc_type, title, supplier_id, version, effective_date, access_scope, "
               f"superseded_by, chunk_no, contains_instructions FROM vec.{self.table} WHERE {' AND '.join(conds)} "
               "ORDER BY id LIMIT %(k)s")
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Document(id=r["id"], page_content=r["content"],
                         metadata={key: r[key] for key in r if key not in ("id", "content")}) for r in rows]


@dataclass
class InMemoryBackend:
    """Unit-test backend: LangChain InMemoryVectorStore (semantic) or a token-
    overlap scorer (keyword).  Same contract, no services needed."""

    docs: list[Document]
    name: str
    vector_store: object | None = None  # InMemoryVectorStore for semantic modes

    def search(self, query, k, scopes, where):
        def ok(d: Document) -> bool:
            return d.metadata["access_scope"] in scopes and all(d.metadata.get(a) == b for a, b in where.items())

        if self.vector_store is not None:
            return self.vector_store.similarity_search(query, k=k, filter=ok)
        terms = set(re.findall(r"\w+", query.lower()))
        scored = [(len(terms & set(re.findall(r"\w+", d.page_content.lower()))), d) for d in self.docs if ok(d)]
        return [d for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0][:k]

    def exact(self, ids, k, scopes, where):
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, ids)) + r")\b", re.IGNORECASE)
        return [d for d in self.docs if d.metadata["access_scope"] in scopes and pattern.search(d.page_content)
                and all(d.metadata.get(a) == b for a, b in where.items())][:k]


# --------------------------------------------------------------------------- service
_COMPONENT_RE = re.compile(r"\bMC-\d+\b", re.IGNORECASE)
_SUPPLIER_RE = re.compile(r"\bS\d{3}\b")


@dataclass
class RetrievalService:
    backends: dict[str, Backend]
    memories: Backend | None = None  # index over approved incident memories
    hybrid_parts: tuple[str, ...] = ("chroma", "keyword")
    graph: object | None = None  # GraphService - enables mode="graph"
    embeddings: object | None = None  # the (cached) query embedder shared by the vector backends
    _fetch_factor: int = field(default=3, repr=False)

    def search(
        self,
        user: User,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        doc_type: str | None = None,
        supplier_id: str | None = None,
        include_superseded: bool = False,
        component_id: str | None = None,
    ) -> list[SearchHit]:
        scopes = sorted(user.document_scopes)
        if not scopes:
            return []
        where = {key: val for key, val in (("doc_type", doc_type), ("supplier_id", supplier_id)) if val}
        fetch = k * self._fetch_factor  # over-fetch: superseded / duplicate chunks are dropped afterwards

        # 1. every mode produces one or more RANKED LISTS ...
        lists: list[tuple[str, list[Document]]] = []
        parts = list(self.hybrid_parts) if mode in ("hybrid", "graph") else [mode]
        for name in parts:
            lists.append((name, self.backends[name].search(query, fetch, scopes, where)))
        ids = sorted({m.group(0).upper() for m in _ID_RE.finditer(query)})
        keyword = self.backends.get("keyword")
        if ids and mode in ("hybrid", "graph") and hasattr(keyword, "exact"):
            # documents that literally contain an identifier from the question
            lists.append(("exact-id", keyword.exact(ids, fetch, scopes, where)))
        if mode == "graph":
            lists += self._graph_lists(query, scopes, where, component_id, supplier_id)

        # 2. ... and ONE fusion step (RRF) combines them, whatever the mode.
        ranked: dict[str, tuple[float, Document, list[str]]] = {}
        for name, docs in lists:
            for rank, doc in enumerate(docs, start=1):
                score, _, seen = ranked.get(doc.id, (0.0, doc, []))
                ranked[doc.id] = (score + 1.0 / (RRF_K + rank), doc, seen + [name])

        order = sorted(ranked.items(), key=lambda kv: -kv[1][0])
        if mode in ("hybrid", "graph"):
            order = self._ids_first(query, order)

        hits = []
        for chunk_id, (score, doc, seen) in order:
            m = doc.metadata
            if not self._allowed(user, m):  # layer 2 of scope enforcement
                continue
            if m.get("superseded_by") and not include_superseded:
                continue
            hits.append(SearchHit(
                chunk_id=chunk_id, source_id=m["source_id"], doc_type=m["doc_type"], title=m["title"],
                supplier_id=m.get("supplier_id") or "", version=int(m.get("version") or 1),
                effective_date=str(m.get("effective_date") or ""), access_scope=m["access_scope"],
                superseded_by=m.get("superseded_by") or "", contains_instructions=bool(m.get("contains_instructions")),
                text=doc.page_content, score=round(score, 5), backends=seen,
            ))
            if len(hits) == k:
                break
        return hits

    @staticmethod
    def _ids_first(query: str, order: list) -> list:
        """Identifiers are exact: if the question contains one (HC-026,
        PO-2026-9001, MC-17 ...), chunks that literally contain it - as a whole
        word, so MC-2 does not match MC-20 - move ahead of chunks that merely
        look similar; the fused order is kept within each group.

        Found by the benchmark: for "incident HC-026" the embedding backend
        cannot tell HC-026 from HC-004, and any-word keyword search scores all
        30 incident reports almost the same (0.078 vs 0.053), so in RRF the
        wrong incidents - found by BOTH lists - outvoted the one exact match.
        """
        ids = {m.group(0).upper() for m in _ID_RE.finditer(query)}
        if not ids:
            return order
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, ids)) + r")\b", re.IGNORECASE)
        exact = [item for item in order if pattern.search(item[1][1].page_content)]
        return exact + [item for item in order if item not in exact]

    # ------------------------------------------------------------ GraphRAG
    def _graph_lists(self, query, scopes, where, component_id, supplier_id) -> list[tuple[str, list[Document]]]:
        """Graph-guided expansion (Chroma + Neo4j hybrid path).

        A query about "MC-17" mentions no supplier, so plain search may never
        surface Cordillera's contract.  The graph knows who supplies MC-17 and
        which contract each SUPPLIES edge cites, so we add:
          * one small search per related supplier (filtered by supplier_id),
          * the chunk of each contract an edge cites (filtered by source_id).
        Every extra list goes through the same backends -> same scope filter.
        """
        if self.graph is None:
            return []
        component_id = component_id or next(iter(_COMPONENT_RE.findall(query)), None)
        supplier_id = supplier_id or next(iter(_SUPPLIER_RE.findall(query)), None)
        if component_id:
            component_id = component_id.upper()
        if not (component_id or supplier_id):
            return []
        related = self.graph.related_for_retrieval(component_id, supplier_id)
        semantic = self.backends[self.hybrid_parts[0]]
        out = []
        for sid in related["suppliers"]:
            if "supplier_id" in where and where["supplier_id"] != sid:
                continue
            out.append((f"graph:{sid}", semantic.search(query, 2, scopes, {**where, "supplier_id": sid})))
        for doc_id in related["documents"]:
            out.append((f"graph-edge:{doc_id}", semantic.search(query, 1, scopes, {**where, "source_id": doc_id})))
        return out

    @staticmethod
    def _allowed(user: User, metadata: dict) -> bool:
        return metadata.get("access_scope") in user.document_scopes

    # ------------------------------------------------------------ memories
    def search_memories(self, user: User, query: str, k: int = 5) -> list[dict]:
        """Semantic search over approved incident memories (scope-checked).

        The memory index is just another Backend (pgvector table
        vec.incident_memories in production), so it gets the same two-layer
        scope enforcement as documents.
        """
        if self.memories is None:
            return []
        docs = self.memories.search(query, k, sorted(user.document_scopes), {})
        return [{**d.metadata, "summary": d.page_content} for d in docs if self._allowed(user, d.metadata)]


# --------------------------------------------------------------------------- factories
def build_postgres_retrieval(pool, embeddings=None, names: dict[str, str] | None = None) -> RetrievalService:
    """Production wiring: pgvector + Chroma + Postgres keyword, memories in pgvector."""
    from supplyguard_ai.llm import get_embeddings
    from supplyguard_ai.retrieval import stores

    n = {"documents": stores.DOCUMENTS_TABLE, "memories": stores.MEMORIES_TABLE,
         "collection": stores.CHROMA_COLLECTION} | (names or {})
    from supplyguard_ai.retrieval.caching import QueryCachingEmbeddings

    emb = QueryCachingEmbeddings(embeddings or get_embeddings())  # graph mode re-uses the query vector
    engine = stores.get_engine()
    return RetrievalService(
        backends={
            "pgvector": PgVectorBackend(stores.documents_pg_store(engine, emb, n["documents"])),
            "chroma": ChromaBackend(stores.documents_chroma_store(stores.chroma_client(), emb, n["collection"])),
            "keyword": PgKeywordBackend(pool, table=n["documents"]),
        },
        memories=PgVectorBackend(stores.memories_pg_store(engine, emb, n["memories"]), name="memories"),
        embeddings=emb,
    )


def build_in_memory_retrieval(documents, memories_rows, embeddings) -> RetrievalService:
    """Unit-test wiring: same chunks, in-memory stores, fake embeddings."""
    from langchain_core.vectorstores import InMemoryVectorStore

    from supplyguard_ai.ingestion.chunking import chunk_all, memory_document

    chunks = chunk_all(documents)
    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(chunks, ids=[c.id for c in chunks])
    mem = InMemoryVectorStore(embeddings)
    mdocs = [memory_document(m) for m in memories_rows]
    mem.add_documents(mdocs, ids=[d.id for d in mdocs])
    return RetrievalService(
        backends={
            "pgvector": InMemoryBackend(chunks, "pgvector", vs),
            "chroma": InMemoryBackend(chunks, "chroma", vs),
            "keyword": InMemoryBackend(chunks, "keyword"),
        },
        memories=InMemoryBackend(mdocs, "memories", mem),
    )
