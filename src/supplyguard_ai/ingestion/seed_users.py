"""Store the app's login users in Postgres, with bcrypt-hashed passwords.

* Users, roles, document scopes and factory access come from
  support-data/config/users.csv (the single source; that file is never edited).
* Each user's password comes from .env: USER_PASSWORD_<USER_ID>, e.g.
  USER_PASSWORD_U_PM_01 for u-pm-01. Only its bcrypt hash is written to the database;
  the plain password is never stored or printed.
* Safe to run again: existing users are updated (role, scopes, password hash).
  To change a password, edit .env and run this script again.

Usage (from the project root):
    uv run python -m supplyguard_ai.ingestion.seed_users
In Docker this runs in the data-load service (see docker-compose.yml).
"""

import csv
import os
import sys
from pathlib import Path

import psycopg

from supplyguard_ai.auth.users import hash_password, normalize
from supplyguard_ai.config import PROJECT_ROOT, postgres_dsn

SUPPORT_DATA_DIR = Path(os.environ.get("SUPPORT_DATA_DIR", PROJECT_ROOT / "support-data"))
USERS_CSV = SUPPORT_DATA_DIR / "config" / "users.csv"
TABLE_SQL = PROJECT_ROOT / "docker" / "postgres" / "02-init_users.sql"


def password_env_name(user_id: str) -> str:
    """u-pm-01 -> USER_PASSWORD_U_PM_01"""
    return "USER_PASSWORD_" + user_id.upper().replace("-", "_")


def read_users() -> list[dict]:
    """The users from users.csv, with list-valued scopes and factories."""
    with open(USERS_CSV, encoding="utf-8", newline="") as f:
        return [
            {
                "user_id": normalize(row["user_id"]),
                "name": row["name"],
                "role": row["role"],
                "scopes": [s for s in row["document_scopes"].split(";") if s],
                "factories": [s for s in row["factory_access"].split(";") if s],
            }
            for row in csv.DictReader(f)
        ]


def seed_users() -> None:
    users = read_users()
    with psycopg.connect(postgres_dsn()) as conn:
        # Same table definition as the Postgres init script, so this also works on a
        # database whose volume existed before the init script was added.
        conn.execute(TABLE_SQL.read_text(encoding="utf-8"))
        for user in users:
            password = os.environ.get(password_env_name(user["user_id"]), "")
            password_hash = hash_password(password) if password else None
            conn.execute(
                """
                INSERT INTO users (user_id, name, role, document_scopes, factory_access, password_hash)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    name            = EXCLUDED.name,
                    role            = EXCLUDED.role,
                    document_scopes = EXCLUDED.document_scopes,
                    factory_access  = EXCLUDED.factory_access,
                    -- no password in .env: keep the hash that is already stored
                    password_hash   = COALESCE(EXCLUDED.password_hash, users.password_hash)
                """,
                (user["user_id"], user["name"], user["role"], user["scopes"], user["factories"],
                 password_hash),
            )
            status = "password set" if password_hash else f"no {password_env_name(user['user_id'])} in .env"
            print(f"seed_users: {user['user_id']:8} {user['role']:22} {status}")
    print(f"seed_users: done ({len(users)} users)")


def main() -> None:
    try:
        seed_users()
    except ValueError as err:  # e.g. a password longer than bcrypt's 72-byte limit
        sys.exit(f"seed_users: {err}")


if __name__ == "__main__":
    main()
