"""Web application for the AI Book Recommendation System.

    python app.py [--db books.db] [--port 5000] [--no-transformer]
"""
from __future__ import annotations

import argparse
import math
import os

from flask import Flask, render_template_string, request

import book_db
from ai_book_recommender import AIBookRecommender

MODES = [("smart", "Smart search"), ("genre", "Genre"), ("author", "Author"),
         ("rating", "Rating"), ("similar", "Similar books"),
         ("letter", "First letter"), ("year", "Year")]
VALID = {m for m, _ in MODES}
ERR_NUM = "Please enter valid numbers."
PLACEHOLDERS = {
    "smart": "e.g. dark fantasy after 1990 rated above 4, or books like Dune",
    "genre": "e.g. Fantasy", "author": "e.g. Tolkien", "similar": "e.g. The Hobbit",
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Book Recommender</title>
<style>
:root{--g1:#eef2ff;--g2:#f5f3ff;--g3:#fdf2f8;--blob1:rgba(99,102,241,.22);--blob2:rgba(236,72,153,.16);--bg:#f4f5f8;--card:rgba(255,255,255,.88);--text:#1d2330;--muted:#667085;--line:#e3e6ec;--accent:#4f46e5;--accent-text:#fff;--err:#b42318}
@media (prefers-color-scheme: dark){:root{--g1:#0b1020;--g2:#1a1740;--g3:#2b1236;--blob1:rgba(129,140,248,.20);--blob2:rgba(236,72,153,.14);--bg:#12141a;--card:rgba(28,31,40,.88);--text:#e8eaf0;--muted:#9aa3b2;--line:#2c303c;--accent:#818cf8;--accent-text:#12141a;--err:#f97066}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--text);background:radial-gradient(circle at 12% 8%,var(--blob1),transparent 45%),radial-gradient(circle at 88% 18%,var(--blob2),transparent 45%),linear-gradient(135deg,var(--g1),var(--g2) 55%,var(--g3));background-attachment:fixed;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.45}
.wrap{max-width:980px;margin:0 auto;padding:20px 14px 40px}
h1{margin:0 0 4px;font-size:1.9rem;background:linear-gradient(90deg,var(--accent),#ec4899);-webkit-background-clip:text;background-clip:text;color:transparent}.sub{color:var(--muted);margin:0 0 18px;font-size:.9rem}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.tabs a{padding:8px 14px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--text);text-decoration:none;font-size:.9rem}
.tabs a.active{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px;box-shadow:0 6px 24px rgba(30,30,80,.08);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px)}
form{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end}
label{display:flex;flex-direction:column;font-size:.8rem;color:var(--muted);gap:4px}
input{padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text);font-size:1rem;min-width:0}
input.grow{width:min(420px,80vw)}input.num{width:100px}
button{padding:10px 18px;border:0;border-radius:8px;background:var(--accent);color:var(--accent-text);font-size:1rem;cursor:pointer}
.err{color:var(--err);font-weight:600}.note{color:var(--muted);font-size:.9rem;margin:0 0 6px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td.t{white-space:normal;min-width:180px}th{color:var(--muted);font-weight:600;font-size:.8rem}
</style></head><body><div class="wrap">
<h1>AI Book Recommender</h1>
<p class="sub">{{ total }} books in database &middot; AI model: {{ model }}</p>
<nav class="tabs">
{% for key, label in modes %}<a href="/?mode={{ key }}" class="{{ 'active' if key == mode }}">{{ label }}</a>{% endfor %}
</nav>
<div class="card">
<form method="get">
<input type="hidden" name="mode" value="{{ mode }}">
{% if mode == 'rating' %}
<label>Minimum rating<input class="num" name="lo" value="{{ lo }}" placeholder="0" inputmode="decimal"></label>
<label>Maximum rating<input class="num" name="hi" value="{{ hi }}" placeholder="5" inputmode="decimal"></label>
{% elif mode == 'year' %}
<label>Year (or start year)<input class="num" name="lo" value="{{ lo }}" placeholder="1990" inputmode="numeric"></label>
<label>End year (optional)<input class="num" name="hi" value="{{ hi }}" placeholder="2000" inputmode="numeric"></label>
{% elif mode == 'letter' %}
<label>First letter<input class="num" name="q" value="{{ q }}" maxlength="1" placeholder="G"></label>
{% else %}
<label>{{ {'smart':'Ask in plain English','genre':'Genre','author':'Author','similar':'Book title'}[mode] }}
<input class="grow" name="q" value="{{ q }}" placeholder="{{ ph }}"></label>
{% endif %}
<label>How many<input class="num" type="number" name="n" min="1" max="500" value="{{ n }}"></label>
<button type="submit">Recommend</button>
</form></div>
{% if error %}<div class="card err">{{ error }}</div>{% endif %}
{% if rows is not none %}
<div class="card">
{% if note %}<p class="note">{{ note }}</p>{% endif %}
{% if rows %}
<p class="note">{{ rows|length }} result{{ '' if rows|length == 1 else 's' }}</p>
<div class="scroll"><table>
<tr><th>#</th><th>Title</th><th>Author</th><th>Genre</th><th>Rating</th><th>Year</th>{% if has_why %}<th>Why</th>{% endif %}</tr>
{% for r in rows %}<tr><td>{{ loop.index }}</td><td class="t">{{ r.title }}</td><td>{{ r.author }}</td><td>{{ r.genre }}</td>
<td>{{ '%.2f' % r.rating }}</td><td>{{ r.year if r.year else '-' }}</td>{% if has_why %}<td>{{ r.why }}</td>{% endif %}</tr>{% endfor %}
</table></div>
{% else %}<p>No books found. Try different input.</p>{% endif %}
</div>{% endif %}
</div></body></html>"""


def _float(s: str) -> float:
    v = float(s)
    if not math.isfinite(v):
        raise ValueError
    return v


def create_app(db_path: str = book_db.DEFAULT_DB, use_transformer: bool = True,
               cache_dir: str | None = "index_cache") -> Flask:
    books = book_db.load_books(db_path)
    if books.empty:
        raise SystemExit(f"{db_path} is empty. Load data first:\n"
                         "  python book_db.py import your_books.csv\n"
                         "  python book_db.py seed        (generated test data)")
    print(f"Building AI index for {len(books)} books ...")
    rec = AIBookRecommender(books, use_transformer, cache_dir=cache_dir)
    total = len(books)
    app = Flask(__name__)

    @app.get("/healthz")  # used by Render to check the service is alive
    def healthz():
        return "ok"

    @app.get("/")
    def index():
        mode = request.args.get("mode", "smart")
        mode = mode if mode in VALID else "smart"
        q = request.args.get("q", "").strip()
        lo = request.args.get("lo", "").strip()
        hi = request.args.get("hi", "").strip()
        n_raw = request.args.get("n", "50").strip() or "50"
        error = note = None
        rows = None
        has_why = False
        n = 50
        if q or lo or hi:
            try:
                n = max(1, min(500, int(n_raw)))
                if mode == "smart":
                    df, note = rec.search(q, n)
                elif mode == "genre":
                    df, note = rec.by_genre(q, n)
                elif mode == "author":
                    df, note = rec.by_author(q, n)
                elif mode == "similar":
                    df, note = rec.similar_to(q, n)
                elif mode == "letter":
                    df, note = rec.by_first_letter(q[:1], n)
                elif mode == "rating":
                    df, note = rec.by_rating(_float(lo) if lo else 0.0,
                                             _float(hi) if hi else 5.0, n)
                else:  # year
                    lo_v = int(lo) if lo else 0
                    hi_v = int(hi) if hi else (lo_v if lo else 9999)
                    df, note = rec.by_year(lo_v, hi_v, n)
                has_why = "why" in df.columns
                rows = df.to_dict("records")
            except ValueError:
                error, note = ERR_NUM, None
        else:
            try:
                n = max(1, min(500, int(n_raw)))
            except ValueError:
                n = 50
        return render_template_string(
            PAGE, modes=MODES, mode=mode, q=q, lo=lo, hi=hi, n=n, error=error,
            note=note, rows=rows, has_why=has_why, total=total,
            model=rec.model_name, ph=PLACEHOLDERS.get(mode, ""))

    app.rec = rec
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=book_db.DEFAULT_DB)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-transformer", action="store_true",
                    help="force the offline TF-IDF model")
    a = ap.parse_args()
    port = int(os.environ.get("PORT", a.port))  # Render supplies PORT
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    create_app(a.db, not a.no_transformer).run(host=host, port=port, debug=False)
