"""Checks that the provided data package is complete.

Every other test and every evaluation result is computed from these files.
If a file is missing or cut short, all later numbers would be wrong,
so we confirm the data is complete before anything else.
"""

from helpers import DATA_DIR, load_csv


def test_main_tables_have_the_expected_number_of_rows():
    # Record counts stated in the assignment (section 06, "Provided data package").
    expected_rows = {
        "structured/suppliers.csv": 60,
        "structured/components.csv": 300,
        "structured/products.csv": 80,
        "structured/purchase_orders.csv": 2000,
        "structured/inventory.csv": 1000,
        "structured/customer_orders.csv": 1500,
    }

    for relative_path, expected in expected_rows.items():
        actual = len(load_csv(relative_path))
        # The message after the comma is shown only if the check fails,
        # so we immediately see which file is wrong.
        assert actual == expected, f"{relative_path}: expected {expected} rows, found {actual}"


def test_every_indexed_document_exists():
    # documents_index.csv lists all 212 documents that will be indexed for retrieval.
    index = load_csv("documents/documents_index.csv")
    assert len(index) == 212

    # Every path in the index must point to a real file, otherwise that
    # document would silently be missing from pgvector and ChromaDB.
    missing = [row["path"] for row in index if not (DATA_DIR / row["path"]).is_file()]
    assert missing == []
