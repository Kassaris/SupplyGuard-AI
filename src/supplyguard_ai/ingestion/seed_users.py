"""
Seed app.users from support-data/config/users.csv, with bcrypt-hashed passwords.

Run by the `data-load` service after load_structured:
    python -m supplyguard_ai.ingestion.seed_users

* Users, roles, document scopes and factory access come from users.csv (the
  single source; that file is never edited).
* The table itself is created by persistence/db.py (ensure_app_schema), the same
  schema the rest of the app reads (services/container.py loads app.users).
* Each user's password comes from .env: USER_PASSWORD_<USER_ID>, e.g.
  USER_PASSWORD_U_PM_01 for u-pm-01. Only its bcrypt hash is written; the plain
  password is never stored or printed.
* UPSERT keeps it idempotent. A user without a password in .env keeps the hash
  already stored. To change a password, edit .env and run this script again.
"""

from __future__ import annotations

import csv
import os
import sys

from dotenv import load_dotenv

from supplyguard_ai.config import PROJECT_ROOT, get_settings
from supplyguard_ai.persistence.db import ensure_app_schema, make_pool
from supplyguard_ai.ui.auth import hash_password

UPSERT = """
INSERT INTO app.users (user_id, name, role, document_scopes, factory_access, password_hash)
VALUES (%(user_id)s, %(name)s, %(role)s, %(scopes)s, %(factories)s, %(password_hash)s)
ON CONFLICT (user_id) DO UPDATE
   SET name = EXCLUDED.name, role = EXCLUDED.role,
       document_scopes = EXCLUDED.document_scopes, factory_access = EXCLUDED.factory_access,
       -- no password in .env: keep the hash that is already stored
       password_hash = COALESCE(EXCLUDED.password_hash, app.users.password_hash)
"""


def password_env_name(user_id: str) -> str:
    """u-pm-01 -> USER_PASSWORD_U_PM_01"""
    return "USER_PASSWORD_" + user_id.upper().replace("-", "_")


def seed(pool, data_dir) -> int:
    """Upsert all users; returns the count. Used by main() and the integration tests."""
    ensure_app_schema(pool)
    with (data_dir / "config" / "users.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with pool.connection() as conn, conn.transaction():
        for r in rows:
            user_id = r["user_id"].strip().lower()
            password = os.environ.get(password_env_name(user_id), "")
            conn.execute(UPSERT, {
                "user_id": user_id, "name": r["name"], "role": r["role"],
                "scopes": [s for s in r["document_scopes"].split(";") if s],
                "factories": [f for f in r["factory_access"].split(";") if f],
                "password_hash": hash_password(password) if password else None,
            })
            status = "password set" if password else f"no {password_env_name(user_id)} in .env (hash kept)"
            print(f"seed_users: {user_id:8} {r['role']:22} {status}")
    return len(rows)


def main() -> int:
    # Settings reads .env into its own fields only; the USER_PASSWORD_* lines must reach
    # os.environ too (Docker already does this through env_file).
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = get_settings()
    pool = make_pool(settings, max_size=1)
    try:
        n = seed(pool, settings.support_data_dir)
    except ValueError as err:  # e.g. a password longer than bcrypt's 72-byte limit
        sys.exit(f"seed_users: {err}")
    finally:
        pool.close()
    print(f"seed_users: done ({n} users)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
