# Deploying to Render

This folder is ready to upload to GitHub and deploy. It contains:

| File / folder | Purpose |
|---|---|
| `app.py`, `ai_book_recommender.py`, `book_db.py`, `build_index.py` | The application |
| `books.db` (22 MB) | All 52,390 books, descriptions removed (they are already baked into the AI index) |
| `index_cache/` (56 MB) | The pre-built AI index, so the server starts in about 2 seconds with low memory |
| `requirements.txt` | Exact library versions the index was built with (do not loosen them) |
| `render.yaml` | Render settings |

## Steps

1. Create an empty repository on GitHub (no README).
2. In this folder, run:
   ```
   git init
   git add .
   git commit -m "AI Book Recommender"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
   git push -u origin main
   ```
   (Files are up to 29 MB each, which is fine for `git push`. The GitHub website's
   drag-and-drop uploader stops at 25 MB per file, so use git or GitHub Desktop.)
3. On render.com: **New > Blueprint**, pick the repository, and click **Apply**.
   (Or **New > Web Service** with build command `pip install -r requirements.txt`
   and start command `gunicorn "app:create_app()" --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`.)
4. Wait for the build to finish and open the `onrender.com` link.

## Notes
- Free plan: the service sleeps after ~15 minutes idle; the first visit afterwards takes about a minute to wake.
- If you change the data in `books.db`, rebuild the index on your own computer with
  `python build_index.py` (full database with descriptions) and commit the new `index_cache/`.
  If the data and index do not match, the app silently falls back to fitting at start-up,
  which needs about 1.5 GB of memory and will fail on the free plan.
