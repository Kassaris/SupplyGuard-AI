"""
Chat history tables for Chainlit's data layer (each user's saved conversations).

WHAT THIS IS
------------
Chainlit can save every conversation ("thread") with its messages ("steps"),
attachments ("elements") and thumbs-up/down ("feedback"), and list a user's past
chats in its sidebar. It does that through its built-in Postgres data layer
(chainlit.data.chainlit_data_layer.ChainlitDataLayer), which is switched on by
the DATABASE_URL environment variable.

Chainlit does NOT create these tables itself, so we do, here. The columns match
the SQL that ChainlitDataLayer runs (Chainlit 2.12): quoted, camelCase names
("User", "threadId", ...) because those are the names its queries use.

WHERE THEY LIVE
---------------
Schema `chat`, separate from our own `app.*` tables (Chainlit's "User" holds only
the login identifier, e.g. u-pm-01; roles and passwords stay in app.users).
Chainlit's queries don't name a schema, so DATABASE_URL sets search_path=chat.

SCHEMA OWNERSHIP (see persistence/db.py)
-----------------------------------------
CHAT_SCHEMA_SQL below -> chat.* tables, idempotent, run at app start-up

TYPES
-----
* ids are TEXT: Chainlit passes uuid strings.
* timestamps are TIMESTAMP(3) without time zone: Chainlit passes naive
  datetime.now() values.
* metadata / props are JSONB: Chainlit writes json.dumps(...) strings and reads
  them back with json.loads(...).
* Foreign keys cascade: deleting a thread (Chainlit's "delete chat") removes its
  steps, elements and feedback. Chainlit always creates the thread and parent
  step before the rows that point to them.
"""

from __future__ import annotations

CHAT_SCHEMA = "chat"

CHAT_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS chat;

-- One row per logged-in user (identifier = our user_id, e.g. u-pm-01).
CREATE TABLE IF NOT EXISTS chat."User" (
    id            text PRIMARY KEY,
    identifier    text NOT NULL UNIQUE,
    metadata      jsonb NOT NULL DEFAULT '{}',
    "createdAt"   timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt"   timestamp(3) NOT NULL DEFAULT now()
);

-- One row per conversation; the sidebar lists a user's threads, newest first.
CREATE TABLE IF NOT EXISTS chat."Thread" (
    id            text PRIMARY KEY,
    "createdAt"   timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt"   timestamp(3) NOT NULL DEFAULT now(),
    "deletedAt"   timestamp(3),
    name          text,
    metadata      jsonb NOT NULL DEFAULT '{}',
    tags          text[] NOT NULL DEFAULT '{}',
    "userId"      text REFERENCES chat."User" (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS thread_user_updated_idx ON chat."Thread" ("userId", "updatedAt" DESC);

-- Messages and steps (user messages, assistant messages, "Working on the case" ...).
CREATE TABLE IF NOT EXISTS chat."Step" (
    id            text PRIMARY KEY,
    "createdAt"   timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt"   timestamp(3) NOT NULL DEFAULT now(),
    "startTime"   timestamp(3),
    "endTime"     timestamp(3),
    "threadId"    text REFERENCES chat."Thread" (id) ON DELETE CASCADE,
    "parentId"    text REFERENCES chat."Step" (id) ON DELETE CASCADE,
    name          text,
    type          text NOT NULL,
    input         text,
    output        text,
    metadata      jsonb NOT NULL DEFAULT '{}',
    "showInput"   text,
    "isError"     boolean DEFAULT false
);
CREATE INDEX IF NOT EXISTS step_thread_idx ON chat."Step" ("threadId", "startTime");

-- Attachments shown in a chat (e.g. the incident_report.md text element).
CREATE TABLE IF NOT EXISTS chat."Element" (
    id              text PRIMARY KEY,
    "createdAt"     timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt"     timestamp(3) NOT NULL DEFAULT now(),
    "threadId"      text REFERENCES chat."Thread" (id) ON DELETE CASCADE,
    "stepId"        text REFERENCES chat."Step" (id) ON DELETE CASCADE,
    metadata        jsonb NOT NULL DEFAULT '{}',
    mime            text,
    name            text,
    "objectKey"     text,
    url             text,
    "chainlitKey"   text,
    display         text,
    size            text,
    language        text,
    page            integer,
    "autoPlay"      boolean,
    "playerConfig"  jsonb,
    props           jsonb DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS element_thread_idx ON chat."Element" ("threadId");

-- Thumbs up / down on a message.
CREATE TABLE IF NOT EXISTS chat."Feedback" (
    id            text PRIMARY KEY,
    "createdAt"   timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt"   timestamp(3) NOT NULL DEFAULT now(),
    "stepId"      text REFERENCES chat."Step" (id) ON DELETE CASCADE,
    name          text NOT NULL,
    value         double precision NOT NULL,
    comment       text
);
CREATE INDEX IF NOT EXISTS feedback_step_idx ON chat."Feedback" ("stepId");
"""


def ensure_chat_history_schema(pool) -> None:
    """Create the chat history tables if missing. Safe to call on every start-up."""
    with pool.connection() as conn:
        # prepare=False: a multi-statement script cannot be a prepared statement
        # (the pool uses prepare_threshold=0, see persistence/db.py).
        conn.execute(CHAT_SCHEMA_SQL, prepare=False)
