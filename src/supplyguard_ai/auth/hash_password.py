"""Print the bcrypt hash of a password, for docker/postgres/02-init_users.sql.

Usage (from the project root):
    uv run python -m supplyguard_ai.auth.hash_password

The password is typed twice and never shown. Copy the printed hash into the SQL file,
in place of the NULL in that user's row.
"""

import getpass
import sys

from supplyguard_ai.auth.users import hash_password


def main() -> None:
    password = getpass.getpass("Password: ")
    if not password:
        sys.exit("The password must not be empty.")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("The passwords do not match.")
    try:
        print(hash_password(password))
    except ValueError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
