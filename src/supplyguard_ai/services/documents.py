"""
Document service: scope-checked access to the 212 provided documents.

WHY THIS EXISTS BEFORE pgvector
-------------------------------
Semantic search (pgvector / ChromaDB) comes in the retrieval step.  But the
agents already need "give me Cordillera's contract" today, which is an exact
lookup by metadata, not a similarity search.  This service:
  * reads `documents/documents_index.csv` (the provenance registry),
  * parses each markdown file's YAML front matter,
  * ENFORCES ACCESS SCOPE IN CODE: a document whose `access_scope` is not
    in the user's scopes is never returned (DOC-PRO-10),
  * flags superseded versions (Atlas contract v2 vs v3).

Later the vector stores will index the same `Document` objects, and their
search results will pass through `filter_for_user()` too.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from supplyguard_ai.domain.models import User


class Document(BaseModel):
    source_id: str
    doc_type: str
    title: str
    supplier_id: str | None
    version: int
    effective_date: str
    access_scope: str
    text: str  # body WITHOUT front matter
    superseded_by: str | None = None  # newer version of the same document, if any


def _strip_front_matter(raw: str) -> str:
    """'---\\n<yaml>\\n---\\n<body>'  ->  '<body>'."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)  # ['', yaml, body]
        if len(parts) == 3:
            return parts[2].strip()
    return raw.strip()


@dataclass
class DocumentService:
    data_dir: Path

    def __post_init__(self) -> None:
        self._docs: dict[str, Document] = {}
        index = self.data_dir / "documents" / "documents_index.csv"
        with index.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            body = _strip_front_matter((self.data_dir / r["path"]).read_text(encoding="utf-8"))
            self._docs[r["source_id"]] = Document(
                source_id=r["source_id"],
                doc_type=r["doc_type"],
                title=r["title"],
                supplier_id=r["supplier_id"] or None,
                version=int(r["version"] or 1),
                effective_date=r["effective_date"],
                access_scope=r["access_scope"],
                text=body,
            )
        self._mark_superseded()

    def _mark_superseded(self) -> None:
        """Same doc_type + supplier, lower version -> superseded by the newest.

        Only applied to contracts: a supplier has ONE contract in force, but
        may legitimately have many notices or e-mails.
        """
        newest: dict[tuple[str, str], Document] = {}
        for d in self._docs.values():
            if d.doc_type != "contract" or not d.supplier_id:
                continue
            key = (d.doc_type, d.supplier_id)
            if key not in newest or d.version > newest[key].version:
                newest[key] = d
        for d in self._docs.values():
            key = (d.doc_type, d.supplier_id or "")
            if key in newest and newest[key].source_id != d.source_id:
                d.superseded_by = newest[key].source_id

    # ------------------------------------------------------------- queries
    @staticmethod
    def filter_for_user(docs: list[Document], user: User) -> list[Document]:
        """THE authorization check for documents.  Every read path calls it."""
        return [d for d in docs if d.access_scope in user.document_scopes]

    def find(
        self,
        user: User,
        supplier_id: str | None = None,
        doc_type: str | None = None,
        include_superseded: bool = True,
    ) -> list[Document]:
        docs = [
            d for d in self._docs.values()
            if (supplier_id is None or d.supplier_id == supplier_id)
            and (doc_type is None or d.doc_type == doc_type)
            and (include_superseded or d.superseded_by is None)
        ]
        # newest first, so the version in force is read first
        docs.sort(key=lambda d: (d.effective_date, d.version), reverse=True)
        return self.filter_for_user(docs, user)

    def get(self, user: User, source_id: str) -> Document | None:
        d = self._docs.get(source_id)
        return d if d and d.access_scope in user.document_scopes else None

    def all_documents(self) -> list[Document]:
        """Unfiltered - ONLY for ingestion scripts (indexing), never for agents."""
        return list(self._docs.values())
