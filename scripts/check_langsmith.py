"""Checks that this machine can send traces to LangSmith.

Run from the repository root:
    uv run python scripts/check_langsmith.py

If everything works, a run named "langsmith_connection_check" appears in the
LangSmith project set in LANGSMITH_PROJECT.
"""

import os
import sys

from dotenv import load_dotenv
from langsmith import Client, traceable

# Use the operating system's certificate store instead of Python's own list, when
# truststore is installed. Needed on networks that inspect HTTPS traffic (corporate
# proxies, antivirus), where the inspecting certificate is trusted by the system
# but not by Python.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# The settings LangSmith needs; all of them are read from the .env file.
REQUIRED_SETTINGS = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]


@traceable(name="langsmith_connection_check")
def connection_check(message: str) -> str:
    """A tiny function whose input and output are recorded as a trace in LangSmith."""
    return f"LangSmith received: {message}"


def main() -> None:
    # Load the variables from .env into the environment of this process.
    load_dotenv()

    # Stop early with a clear message if a setting is missing.
    missing = [name for name in REQUIRED_SETTINGS if not os.getenv(name)]
    if missing:
        sys.exit(f"Missing settings in .env: {', '.join(missing)}")

    # Calling the API once tells us whether the key and the endpoint are valid.
    client = Client()
    try:
        next(iter(client.list_projects(limit=1)), None)
    except Exception as error:
        sys.exit(f"Could not connect to LangSmith: {error}")

    # Send the trace through the same client, so flush() below waits for it.
    print(connection_check("hello from SupplyGuard AI", langsmith_extra={"client": client}))

    # Traces are sent in the background; wait until they have been uploaded.
    client.flush()
    print(f"Trace sent. Open project '{os.getenv('LANGSMITH_PROJECT')}' in LangSmith to see it.")


if __name__ == "__main__":
    main()
