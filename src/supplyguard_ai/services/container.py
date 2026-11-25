"""
Composition root for the application service layer.

WHY
---
LangGraph nodes, LangChain tools, the MCP servers and Chainlit all need the
same services wired the same way (same audit log, same approval store).
Instead of each one constructing its own objects, they all receive ONE
`AppServices` instance.

TWO STORAGE BACKENDS, ONE WIRING (settings.storage_backend)
----------------------------------------------------------
                      memory (unit tests)        postgres (app / docker compose)
  operational data    CsvRepository              PostgresRepository  (ops.*)
  users               users.csv                  app.users
  audit trail         AuditLog                   PostgresAuditLog    (app.audit_events)
  approvals           InMemoryApprovalStore      PostgresApprovalStore (app.approvals)
  procurement ledger  InMemoryLedger             PostgresLedger      (app.procurement_executions)
  graph checkpoints   InMemorySaver              PostgresSaver       (LangGraph tables)

Business services are identical in both columns - they only see protocols.
The LangGraph graph gets this object through its runtime *context*, so nodes
and tools use services without globals and without the LLM seeing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from supplyguard_ai.config import Settings, get_settings
from supplyguard_ai.services.approvals import ApprovalService
from supplyguard_ai.services.audit import AuditLog, AuditStore
from supplyguard_ai.services.documents import DocumentService
from supplyguard_ai.services.identity import IdentityService
from supplyguard_ai.services.impact import ImpactService
from supplyguard_ai.services.procurement import SimulatedProcurementSystem
from supplyguard_ai.services.recovery import RecoveryPlanner
from supplyguard_ai.services.repository import CsvRepository


@dataclass
class AppServices:
    repo: CsvRepository
    identity: IdentityService
    documents: DocumentService
    impact: ImpactService
    planner: RecoveryPlanner
    audit: AuditStore
    approvals: ApprovalService
    procurement: SimulatedProcurementSystem
    pool: object | None = field(default=None, repr=False)  # psycopg ConnectionPool in postgres mode
    settings: Settings | None = field(default=None, repr=False)
    _cases: object | None = field(default=None, repr=False)

    @property
    def cases(self):
        """Case registry (services/cases.py): app.cases in postgres mode, memory otherwise."""
        if self._cases is None:
            from supplyguard_ai.services.cases import InMemoryCaseRegistry, PostgresCaseRegistry

            self._cases = PostgresCaseRegistry(self.pool) if self.pool is not None else InMemoryCaseRegistry()
        return self._cases

    @property
    def risk_feed(self):
        """External risk alerts.
          EXTERNAL_MCP_URL set (docker compose) -> external MCP server -> external_info_service
          not set                               -> the same risk_feed.json, read locally
        """
        from supplyguard_ai.mcp_client import FileRiskFeed, McpRiskFeed

        url = self.settings.external_mcp_url if self.settings else None
        return McpRiskFeed(url) if url else FileRiskFeed(self.repo.data_dir / "external_service" / "risk_feed.json")

    @property
    def procurement_gateway(self):
        """How the workflow executes approved actions.
          PROCUREMENT_MCP_URL set (docker compose) -> through the procurement MCP server
          not set (unit tests, local dev)          -> in-process call, same rules
        """
        from supplyguard_ai.mcp_client import InProcessGateway, McpProcurementGateway

        url = self.settings.procurement_mcp_url if self.settings else None
        return McpProcurementGateway(url) if url else InProcessGateway(self.procurement)
    _retrieval: object | None = field(default=None, repr=False)
    _graph: object | None = field(default=None, repr=False)

    @property
    def graph(self):
        """Dependency graph (services/graph.py GraphService), built on first use.
          memory mode   -> InMemoryGraph from support-data/graph/*.csv
          postgres mode -> Neo4jGraph (bolt, read transactions; seeded by neo4j-seed)
        """
        if self._graph is None:
            from supplyguard_ai.services.graph import GraphService, InMemoryGraph

            if self.pool is None:
                store = InMemoryGraph(self.repo.data_dir / "graph")
            else:
                from supplyguard_ai.persistence.neo4j_graph import Neo4jGraph

                store = Neo4jGraph.from_settings()
            self._graph = GraphService(store)
        return self._graph

    @property
    def retrieval(self):
        """Document / memory search (retrieval/service.py), built on first use.

        Lazy because it needs the embedding model (and in postgres mode Chroma):
        code paths that never search - loaders, approvals, unit tests of the
        planner - must not depend on those services being reachable.
          memory mode   -> in-memory index over the same chunks, fake embeddings (no network)
          postgres mode -> pgvector + ChromaDB + Postgres full-text (indexed by doc-ingest)
        """
        if self._retrieval is None:
            from supplyguard_ai.retrieval.service import build_in_memory_retrieval, build_postgres_retrieval

            if self.pool is None:
                from langchain_core.embeddings import DeterministicFakeEmbedding

                from supplyguard_ai.llm import EMBEDDING_DIM

                self._retrieval = build_in_memory_retrieval(
                    self.documents.all_documents(), self.repo.approved_memories(), DeterministicFakeEmbedding(size=EMBEDDING_DIM))
            else:
                self._retrieval = build_postgres_retrieval(self.pool)
            self._retrieval.graph = self.graph  # enables mode="graph" (Chroma + Neo4j hybrid path)
        return self._retrieval

    @classmethod
    def from_settings(cls, settings: Settings | None = None, pool=None) -> "AppServices":
        s = settings or get_settings()
        if s.storage_backend == "postgres":
            return cls._postgres(s, pool)
        return cls._memory(s)

    # ------------------------------------------------------------------ memory
    @classmethod
    def _memory(cls, s: Settings) -> "AppServices":
        repo = CsvRepository(s.support_data_dir)
        return cls._assemble(s, repo, IdentityService.from_csv(s.support_data_dir / "config" / "users.csv"),
                             AuditLog(), approval_store=None, ledger=None, pool=None)

    # ---------------------------------------------------------------- postgres
    @classmethod
    def _postgres(cls, s: Settings, pool=None) -> "AppServices":
        # Imported here so unit tests never need psycopg / a database.
        from supplyguard_ai.persistence.db import ensure_app_schema, get_pool
        from supplyguard_ai.persistence.repository import PostgresRepository
        from supplyguard_ai.persistence.stores import PostgresApprovalStore, PostgresAuditLog, PostgresLedger

        pool = pool or get_pool()
        ensure_app_schema(pool)
        with pool.connection() as conn:
            users = conn.execute("SELECT * FROM app.users").fetchall()
        if not users:
            raise RuntimeError("app.users is empty - run: python -m supplyguard_ai.ingestion.seed_users")
        return cls._assemble(s, PostgresRepository(s.support_data_dir, pool), IdentityService.from_rows(users),
                             PostgresAuditLog(pool), approval_store=PostgresApprovalStore(pool),
                             ledger=PostgresLedger(pool), pool=pool)

    # ------------------------------------------------------------------ shared
    @classmethod
    def _assemble(cls, s, repo, identity, audit, approval_store, ledger, pool) -> "AppServices":
        policies = repo.policies()
        approvals = ApprovalService(tiers=policies["approval_tiers_eur"], audit=audit)
        if approval_store is not None:
            approvals.store = approval_store
        procurement = SimulatedProcurementSystem(
            repo=repo, approvals=approvals, audit=audit, spending_cap_eur=policies["incident_spending_cap_eur"],
        )
        if ledger is not None:
            procurement.ledger = ledger
        return cls(
            repo=repo, identity=identity, documents=DocumentService(s.support_data_dir),
            impact=ImpactService(repo), planner=RecoveryPlanner(repo), audit=audit,
            approvals=approvals, procurement=procurement, pool=pool, settings=s,
        )

    # ------------------------------------------------------------- checkpoints
    def make_checkpointer(self):
        """LangGraph checkpointer matching the storage backend.

        Postgres: state of every case is saved after every node, so a case
        survives an app restart and can be resumed days later by thread_id.
        """
        if self.pool is None:
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver(self.pool)
        saver.setup()  # creates/migrates the checkpoint tables (idempotent)
        return saver
