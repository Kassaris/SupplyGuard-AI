"""
Documents -> LangChain `Document` chunks with full provenance (Week 3, day 12:
"Document loaders and text splitters").

PIPELINE
--------
  DocumentService (parses front matter, marks superseded, knows access scope)
        │  one Document per file
        ▼
  RecursiveCharacterTextSplitter (800 chars, 100 overlap)
        │  1-3 chunks per document (all files are < 1.2 KB)
        ▼
  contextual header + metadata + deterministic id
        ▼
  pgvector table  /  Chroma collection

DESIGN DECISIONS
----------------
* Reuse DocumentService instead of a generic loader: it already parses the
  front matter and computes `superseded_by`; a second parser could disagree.
* RecursiveCharacterTextSplitter: splits on paragraph, then line, then
  sentence boundaries, so a contract clause is rarely cut in half.  Chunk
  size 800 keeps one or two clauses per chunk = focused embeddings;
  100 overlap keeps a clause that straddles the cut retrievable from both sides.
* CONTEXTUAL HEADER: each chunk starts with "<title> | <doc_type> | <supplier>".
  The second chunk of a contract otherwise has no supplier name in it, and a
  query like "Cordillera expedite fee" could miss it.
* DETERMINISTIC IDS "<source_id>#<chunk_no>": re-running ingestion UPSERTS the
  same rows instead of adding duplicates (idempotent, like the compose job).
* PROVENANCE IN METADATA (assignment §06): source_id, version, effective_date,
  access_scope, superseded_by travel with every chunk, so every retrieved
  claim can be cited and scope-filtered.
* SECURITY FLAG AT INGESTION: `contains_instructions` is computed once with
  the deterministic detector, so search results can warn before the model
  even reads the text.
* Metadata values are scalars only (str/int/bool) - Chroma rejects None and
  lists; empty string means "none".
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from supplyguard_ai.core.prompts import detect_injection
from supplyguard_ai.services.documents import Document as SourceDocument

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Columns every vector backend stores and can filter on.
METADATA_FIELDS = (
    "source_id", "doc_type", "title", "supplier_id", "version", "effective_date",
    "access_scope", "superseded_by", "chunk_no", "contains_instructions",
)


def chunk_document(doc: SourceDocument) -> list[Document]:
    """One source document -> its chunks (LangChain Documents with ids)."""
    header = f"{doc.title} | {doc.doc_type} | supplier {doc.supplier_id or '-'}"
    flagged = bool(detect_injection(doc.text))
    chunks = []
    for i, piece in enumerate(SPLITTER.split_text(doc.text)):
        chunks.append(Document(
            id=f"{doc.source_id}#{i}",
            page_content=f"{header}\n{piece}",
            metadata={
                "source_id": doc.source_id,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "supplier_id": doc.supplier_id or "",
                "version": doc.version,
                "effective_date": doc.effective_date,
                "access_scope": doc.access_scope,
                "superseded_by": doc.superseded_by or "",
                "chunk_no": i,
                "contains_instructions": flagged,
            },
        ))
    return chunks


def chunk_all(docs: list[SourceDocument]) -> list[Document]:
    return [c for d in docs for c in chunk_document(d)]


# --------------------------------------------------------------------------- memories
MEMORY_FIELDS = ("memory_id", "case_id", "component_id", "supplier_id", "strategy", "access_scope")


def memory_document(m: dict) -> Document:
    """An approved incident memory -> one Document (no splitting: one sentence each).

    The text that gets embedded is the approved summary plus the strategy, so a
    search like "upstream shortage on a chip, what worked?" finds it by meaning.
    """
    return Document(
        id=m["memory_id"],
        page_content=f"{m['approved_summary']} Strategy: {m['strategy']}.",
        metadata={k: str(m.get(k) or "") for k in MEMORY_FIELDS},
    )
