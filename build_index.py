"""Build the AI index once (needs ~2 GB RAM, about a minute), so the web app can start fast.

    python build_index.py [--db books.db] [--out index_cache]
"""
import argparse
import time

import book_db
from ai_book_recommender import AIBookRecommender, save_index

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=book_db.DEFAULT_DB)
ap.add_argument("--out", default="index_cache")
a = ap.parse_args()
books = book_db.load_books(a.db)
print(f"Fitting the AI index for {len(books)} books ...")
t = time.time()
rec = AIBookRecommender(books, use_transformer=False)
save_index(rec, a.out)
print(f"Saved to {a.out}/ in {time.time() - t:.0f}s")
