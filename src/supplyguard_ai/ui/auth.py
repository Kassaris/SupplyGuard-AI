"""
Login for the Chainlit password auth: the user's name + personal password.

WHO THE USERS ARE
-----------------
`services.identity` (services/identity.py) holds the users from app.users
(seeded by ingestion/seed_users.py from users.csv). The login form takes the
user's NAME ("Maria Keller"), matched without regard to case or extra spaces;
the returned domain `User` carries the user_id, role and document scopes that
the rest of the app uses.

PASSWORDS
---------
bcrypt: salted and deliberately slow. Only the hash is stored (app.users.
password_hash); the plain password lives in .env as USER_PASSWORD_<USER_ID>
and is hashed by seed_users.

* postgres mode: the hash is read from app.users.
* memory mode (unit tests / local dev without Postgres): the password is
  compared with USER_PASSWORD_<USER_ID> from the environment.

Unknown name and wrong password give the same answer (None) and take about the
same time, so the login form does not reveal which users exist.
"""

from __future__ import annotations

import hmac
import os

import bcrypt

# bcrypt only looks at the first 72 bytes of a password, and bcrypt 5 raises an error above that.
MAX_PASSWORD_BYTES = 72

# A real bcrypt hash of a random value: checked for unknown names, so they cost the same time.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt()).decode()


def normalize(name: str) -> str:
    """One spelling per user: "  Maria   KELLER " and "maria keller" are the same login."""
    return " ".join(name.split()).lower()


def hash_password(password: str) -> str:
    """Return the bcrypt hash to store. A new random salt is built into every hash."""
    data = password.encode("utf-8")
    if len(data) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password is longer than {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(data, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, stored: str | None) -> bool:
    """True if the password matches the stored bcrypt hash."""
    data = password.encode("utf-8")
    if not stored or len(data) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(data, stored.encode("utf-8"))
    except ValueError:  # not a bcrypt hash
        return False


def password_env_name(user_id: str) -> str:
    """u-pm-01 -> USER_PASSWORD_U_PM_01"""
    return "USER_PASSWORD_" + user_id.upper().replace("-", "_")


# --------------------------------------------------------------------------- password stores
class PostgresPasswords:
    def __init__(self, pool) -> None:
        self.pool = pool

    def check(self, user_id: str, password: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT password_hash FROM app.users WHERE user_id = %s", (user_id,)).fetchone()
        return verify_password(password, row["password_hash"] if row else None)


class EnvPasswords:
    """Memory mode: compare with USER_PASSWORD_<USER_ID> from the environment."""

    def check(self, user_id: str, password: str) -> bool:
        expected = os.environ.get(password_env_name(user_id))
        if not expected:
            return False
        return hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8"))


# --------------------------------------------------------------------------- login
def find_user(services, name: str):
    """The domain User whose name matches the typed login name, or None."""
    wanted = normalize(name)
    return next((u for u in services.identity.all_users() if normalize(u.name) == wanted), None)


def authenticate(services, username: str, password: str):
    """Return the domain User for a valid name + password, else None.

    Synchronous (database + bcrypt): the Chainlit callback runs it in a thread.
    """
    user = find_user(services, username)
    if user is None:
        verify_password(password, _DUMMY_HASH)  # spend similar time either way
        return None
    store = PostgresPasswords(services.pool) if services.pool is not None else EnvPasswords()
    return user if store.check(user.user_id, password) else None
