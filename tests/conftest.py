"""Shared pytest setup."""

import os

# Some antivirus tools on Windows set SSLKEYLOGFILE to a path that crashes the
# Postgres and Neo4j drivers. The tests don't need it.
os.environ.pop("SSLKEYLOGFILE", None)
