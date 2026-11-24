"""User accounts: password hashing and the login check against the Postgres `users` table
(created by docker/postgres/02-init_users.sql, filled by supplyguard_ai.ingestion.seed_users).

Users log in with their name (e.g. "Maria Keller") and password. The name is matched
without regard to upper/lower case or extra spaces.
"""

import asyncio

import bcrypt
import psycopg

from supplyguard_ai.config import postgres_dsn

# bcrypt only looks at the first 72 bytes of a password, and bcrypt 5 raises an error above that.
MAX_PASSWORD_BYTES = 72


def normalize(username: str) -> str:
    """One spelling per user: "  Maria   KELLER " and "maria keller" are the same login.
    Trims, collapses inner spaces and lowercases."""
    return " ".join(username.split()).lower()


def hash_password(password: str) -> str:
    """Return the bcrypt hash to store. A new random salt is built into every hash."""
    data = password.encode("utf-8")
    if len(data) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password is longer than {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(data, bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    """True if the password matches the stored hash."""
    data = password.encode("utf-8")
    if len(data) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(data, password_hash.encode("utf-8"))


def _find_user(login_name: str) -> tuple | None:
    """Read the users row whose name matches the (normalized) login name, or None.
    The name is compared the same way normalize() spells it: trimmed, single spaces, lowercase."""
    with psycopg.connect(postgres_dsn()) as conn:
        return conn.execute(
            "SELECT user_id, name, role, document_scopes, factory_access, password_hash "
            "FROM users WHERE lower(regexp_replace(btrim(name), '\\s+', ' ', 'g')) = %s",
            (login_name,),
        ).fetchone()


async def authenticate(username: str, password: str) -> dict | None:
    """Log in with the user's name (e.g. "Maria Keller") and password.

    Return the user's identity, role and access for a correct login, otherwise None:
    {"user_id", "name", "role", "scopes", "factories"}. user_id stays the internal key
    (sessions, saved conversations); the name is only what the user types.

    The role decides what the user may approve; scopes decide which documents retrieval
    may return to them; factories are the plants they work with.
    """
    # The lookup uses a normal (synchronous) connection in a worker thread: psycopg's async
    # mode does not run on Windows' default event loop, and this works on every OS.
    row = await asyncio.to_thread(_find_user, normalize(username))

    if row is None:
        return None
    user_id, name, role, scopes, factories, password_hash = row
    if password_hash is None:
        # No password set yet for this user (see 02-init_users.sql): no login possible.
        return None

    # bcrypt is deliberately slow (~0.2 s), so run it in a thread to keep the app responsive.
    if not await asyncio.to_thread(check_password, password, password_hash):
        return None
    return {
        "user_id": user_id,
        "name": name,
        "role": role,
        "scopes": list(scopes),
        "factories": list(factories),
    }
