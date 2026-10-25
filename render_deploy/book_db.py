"""Database layer: SQLite schema, CSV importer and loader.

Command line:
    python book_db.py import books.csv [--db books.db]
    python book_db.py seed [--n 50000] [--db books.db]   # generated test data
    python book_db.py count [--db books.db]
    python book_db.py export-sql books.sql [--db books.db] [--top 2000]   # DB (or top-N sample) as SQL
    python book_db.py load-sql books.sql [--db books.db]     # build a DB from a SQL script
"""
from __future__ import annotations

import argparse
import ast
import re
import sqlite3
from datetime import date

import pandas as pd

DEFAULT_DB = "books.db"
CHUNK = 50_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    author        TEXT,
    genre         TEXT,
    rating        REAL DEFAULT 0,
    ratings_count INTEGER DEFAULT 1,
    year          INTEGER DEFAULT 0,
    description   TEXT,
    UNIQUE (title, author)
);
CREATE INDEX IF NOT EXISTS idx_books_title  ON books(title);
CREATE INDEX IF NOT EXISTS idx_books_author ON books(author);
CREATE INDEX IF NOT EXISTS idx_books_genre  ON books(genre);
CREATE INDEX IF NOT EXISTS idx_books_rating ON books(rating);
CREATE INDEX IF NOT EXISTS idx_books_year   ON books(year);
"""

# Header aliases (earlier = higher priority). Compared after lower-casing and
# removing spaces, underscores and hyphens.
ALIASES = {
    "title": ["title", "booktitle", "bookname", "name"],
    "author": ["author", "authors", "authorname", "writer"],
    "genre": ["genres", "genre", "categories", "category", "shelves"],
    "rating": ["rating", "averagerating", "avgrating", "avg"],
    "ratings_count": ["numratings", "ratingscount", "ratingcount", "numberofratings", "ratingsnumber", "ratings"],
    "year": ["originalpublicationyear", "firstpublishdate", "publicationyear", "year",
             "publishdate", "publicationdate", "publisheddate", "published", "date"],
    "description": ["description", "summary", "synopsis", "desc", "about"],
}


def _norm(h: str) -> str:
    return re.sub(r"[\s_\-]", "", str(h).lower())


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def insert_books(conn: sqlite3.Connection, rows) -> int:
    """Insert dict rows; duplicates (title, author) are ignored. Returns rows added."""
    data = [(r["title"], r.get("author") or "Unknown", r.get("genre") or "",
             float(r.get("rating") or 0), int(r.get("ratings_count") or 1),
             int(r.get("year") or 0), r.get("description") or "") for r in rows]
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO books (title, author, genre, rating, ratings_count, year, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    conn.commit()
    return conn.total_changes - before


# ---- cleaning ---------------------------------------------------------------
def _clean_genre(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, (list, tuple)):
                return ", ".join(str(x).strip() for x in items if str(x).strip())
        except (ValueError, SyntaxError):
            items = re.findall(r"""['"]([^'"]+)['"]""", s)
            return ", ".join(x.strip() for x in items if x.strip())
    return ", ".join(x.strip() for x in re.split(r"[;,|]", s) if x.strip())


_MAX_YEAR = date.today().year + 1


def _clean_year(v) -> int:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    s = str(v)
    m = re.search(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)", s)
    if m:
        y = int(m.group(1))
    else:  # mm/dd/yy
        m = re.search(r"(?<!\d)\d{1,2}/\d{1,2}/(\d{2})(?!\d)", s)
        if not m:
            return 0
        yy = int(m.group(1))
        y = 2000 + yy if yy <= _MAX_YEAR % 100 else 1900 + yy
    return y if 0 < y <= _MAX_YEAR else 0


def _clean_chunk(chunk: pd.DataFrame, colmap: dict) -> list[dict]:
    out = pd.DataFrame({k: chunk[src] for k, src in colmap.items() if k != "year"})
    # Year: try every matching date column in priority order, first usable value wins
    year = pd.Series(0, index=chunk.index)
    for col in colmap.get("year", []):
        y = chunk[col].map(_clean_year)
        year = year.where(year > 0, y)
    out["year"] = year
    for c in ALIASES:
        if c not in out:
            out[c] = None
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out = out[out["title"] != ""].copy()
    out["author"] = (out["author"].fillna("").astype(str)
                     .str.replace(r"\s*\(Goodreads Author\)", "", regex=True)
                     .str.strip().replace("", "Unknown"))
    out["genre"] = out["genre"].map(_clean_genre)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(0).clip(0, 5)
    cnt = out["ratings_count"].astype(str).str.replace(r"[,\s]", "", regex=True)
    out["ratings_count"] = pd.to_numeric(cnt, errors="coerce").fillna(1).clip(lower=1).astype(int)
    out["description"] = out["description"].fillna("").astype(str)
    return out[list(ALIASES)].to_dict("records")


def import_csv(csv_path: str, db_path: str = DEFAULT_DB) -> dict:
    """Import a CSV in chunks. Returns {'read': n, 'added': n}."""
    conn = connect(db_path)
    read = added = 0
    colmap = None
    reader = pd.read_csv(csv_path, chunksize=CHUNK, dtype=str, on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    for chunk in reader:
        if colmap is None:
            headers = {_norm(c): c for c in chunk.columns}
            colmap = {}
            for field, names in ALIASES.items():
                found = [headers[a] for a in names if a in headers]
                if found:
                    colmap[field] = found if field == "year" else found[0]
            if "title" not in colmap:
                raise ValueError(f"No title column found. Headers: {list(chunk.columns)}")
        rows = _clean_chunk(chunk, colmap)
        read += len(chunk)
        added += insert_books(conn, rows)
    conn.close()
    return {"read": read, "added": added, "columns": colmap}


# ---- reading ---------------------------------------------------------------
def load_books(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM books", conn)
    finally:
        conn.close()


def count(db_path: str = DEFAULT_DB) -> int:
    conn = connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    finally:
        conn.close()


def seed(n: int = 50_000, db_path: str = DEFAULT_DB) -> int:
    from ai_book_recommender import synthetic_books
    conn = connect(db_path)
    added = insert_books(conn, synthetic_books(n).to_dict("records"))
    conn.close()
    return added


def export_sql(out_path: str, db_path: str = DEFAULT_DB, top: int | None = None) -> int:
    """Write the database (schema + INSERT statements) to a .sql file.

    top=N exports only the N most-rated books, a small real sample for GitHub.
    """
    src = connect(db_path)
    if top:
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        rows = src.execute(
            "SELECT title, author, genre, rating, ratings_count, year, description "
            "FROM books ORDER BY ratings_count DESC LIMIT ?", (top,)).fetchall()
        conn.executemany(
            "INSERT INTO books (title, author, genre, rating, ratings_count, year, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    else:
        conn = src
    with open(out_path, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(line + "\n")
    n = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    src.close()
    if top:
        conn.close()
    return n


def load_sql(sql_path: str, db_path: str = DEFAULT_DB) -> int:
    """Run a SQL script (such as real_books.sql) against the database."""
    conn = sqlite3.connect(db_path)
    with open(sql_path, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return count(db_path)


def main():
    ap = argparse.ArgumentParser(description="Book database tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import"); p.add_argument("csv"); p.add_argument("--db", default=DEFAULT_DB)
    p = sub.add_parser("seed"); p.add_argument("--n", type=int, default=50_000); p.add_argument("--db", default=DEFAULT_DB)
    p = sub.add_parser("count"); p.add_argument("--db", default=DEFAULT_DB)
    p = sub.add_parser("export-sql"); p.add_argument("sql"); p.add_argument("--db", default=DEFAULT_DB); p.add_argument("--top", type=int)
    p = sub.add_parser("load-sql"); p.add_argument("sql"); p.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args()
    if a.cmd == "import":
        r = import_csv(a.csv, a.db)
        print(f"Read {r['read']} rows, added {r['added']} new books. Columns used: {r['columns']}")
    elif a.cmd == "seed":
        print(f"Added {seed(a.n, a.db)} generated books.")
    elif a.cmd == "export-sql":
        print(f"Exported {export_sql(a.sql, a.db, a.top)} books to {a.sql}")
    elif a.cmd == "load-sql":
        load_sql(a.sql, a.db)
    print(f"Books in {a.db}: {count(a.db)}")


if __name__ == "__main__":
    main()
