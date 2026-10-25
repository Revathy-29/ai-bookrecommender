"""AI engine for the Book Recommendation System.

Embeds every book as a vector, finds neighbours with one matrix-vector
product, and blends similarity with a Bayesian "quality" score.

Run `python ai_book_recommender.py` for a small text interface.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SIM_W = 0.75     # weight of semantic similarity in final score
QUAL_W = 0.25    # weight of Bayesian quality in final score
COLUMNS = ["title", "author", "genre", "rating", "year"]


# --------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------
class Embedder:
    """Turns text into L2-normalised vectors.

    Uses sentence-transformers (all-MiniLM-L6-v2) when installed, otherwise
    TF-IDF + TruncatedSVD (LSA), which works offline on any CPU.
    """

    def __init__(self, use_transformer: bool = True, dims: int = 256):
        self.dims = dims
        self.st = None
        self.name = "TF-IDF + SVD (LSA)"
        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer("all-MiniLM-L6-v2")
                self.name = "sentence-transformers all-MiniLM-L6-v2 (384-d)"
            except Exception:
                self.st = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.st is not None:
            return self._st_encode(texts)
        n = len(texts)
        self.tfidf = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
            min_df=2 if n >= 1000 else 1, max_features=60000)
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        Z = normalize(self.svd.fit_transform(X)).astype(np.float32)
        self.svd.components_ = self.svd.components_.astype(np.float32)  # halves memory
        del X
        return Z

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.st is not None:
            return self._st_encode(texts)
        return normalize(self.svd.transform(self.tfidf.transform(texts))).astype(np.float32)

    def _st_encode(self, texts):
        return np.asarray(self.st.encode(
            texts, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False), dtype=np.float32)


class _CachedSVD:
    """Minimal stand-in for a fitted TruncatedSVD (only .transform is needed)."""
    def __init__(self, components):
        self.components_ = components

    def transform(self, X):
        return np.asarray(X @ self.components_.T)


def _titles_hash(titles) -> str:
    return hashlib.md5("\n".join(map(str, titles)).encode("utf-8")).hexdigest()


def save_index(rec, folder: str) -> None:
    """Save the fitted LSA index (arrays only, no pickles) so the app can start in seconds
    without re-fitting. Stored as float16 to keep the files small."""
    emb = rec.embedder
    if emb.st is not None:
        raise ValueError("Only the TF-IDF + SVD model can be cached.")
    os.makedirs(folder, exist_ok=True)
    vocab = emb.tfidf.vocabulary_
    terms = np.empty(len(vocab), dtype=object)
    for t, i in vocab.items():
        terms[i] = t
    np.save(os.path.join(folder, "emb.npy"), rec.emb.astype(np.float16))
    np.savez_compressed(os.path.join(folder, "lsa.npz"), terms=terms.astype(str),
                        idf=emb.tfidf.idf_.astype(np.float32),
                        comps=emb.svd.components_.astype(np.float16))
    meta = {"n": rec.n, "titles_md5": _titles_hash(rec.df["title"]),
            "ngram_range": list(emb.tfidf.ngram_range), "dims": int(rec.emb.shape[1])}
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_index(folder, titles):
    """Return (embedder, emb) from a saved index, or None if missing / not matching the data."""
    try:
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        if meta["n"] != len(titles) or meta["titles_md5"] != _titles_hash(titles):
            return None
        emb = np.load(os.path.join(folder, "emb.npy")).astype(np.float32)
        z = np.load(os.path.join(folder, "lsa.npz"))
    except (OSError, KeyError, ValueError):
        return None
    e = Embedder(use_transformer=False, dims=meta["dims"])
    vocab = {t: i for i, t in enumerate(z["terms"].tolist())}
    e.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                              ngram_range=tuple(meta["ngram_range"]), vocabulary=vocab,
                              dtype=np.float32)
    e.tfidf.fit(["warm up"])          # nothing is learned: the vocabulary is fixed
    e.tfidf.idf_ = z["idf"].astype(np.float32)
    e.svd = _CachedSVD(z["comps"].astype(np.float32))
    return e, emb


# --------------------------------------------------------------------------
# Natural-language query parser
# --------------------------------------------------------------------------
_YEAR = r"(1\d{3}|20\d{2})"
_NUM = r"([0-5](?:\.\d+)?)"
_STOP = r"(?:(?:after|since|before|until|till|published|rated|rating|starting|starts|between|by|from|with|at least|above|over)\b|in\s+\d)"
_FILLER = (r"\b(books?|novels?|show me|find me|find|recommend|give me|suggest|"
           r"some|please|i want|i'd like|published|that|are|is)\b")


def parse_query(text: str) -> dict:
    """Extract structured conditions; whatever is left is the semantic query."""
    t = " " + text.lower().strip() + " "
    out = {"year_lo": None, "year_hi": None, "rating_min": None,
           "letter": None, "author": None, "like": None, "rest": ""}

    m = re.search(r"\b(?:like|similar to)\s+(.+?)(?=\s+" + _STOP + r"|\s*$)", t)
    if m:
        out["like"] = m.group(1).strip(" .,'\"")
        t = t[:m.start()] + " " + t[m.end():]

    def take(pattern, fn):
        nonlocal t
        m = re.search(pattern, t)
        if m:
            fn(m)
            t = t[:m.start()] + " " + t[m.end():]
            return True
        return False

    def yr(lo=None, hi=None):
        def f(m):
            if lo is not None:
                out["year_lo"] = int(m.group(lo))
            if hi is not None:
                out["year_hi"] = int(m.group(hi))
        return f

    (take(r"\bbetween\s+" + _YEAR + r"\s+and\s+" + _YEAR, yr(1, 2))
     or take(r"\b" + _YEAR + r"\s*(?:-|to)\s*" + _YEAR + r"\b", yr(1, 2)))
    take(r"\b(?:after|since|from)\s+" + _YEAR + r"\b", yr(lo=1))
    take(r"\b(?:before|until|till)\s+" + _YEAR + r"\b", yr(hi=1))
    take(r"\b(?:published\s+)?in\s+" + _YEAR + r"\b", lambda m: out.update(
        year_lo=int(m.group(1)), year_hi=int(m.group(1))))

    setr = lambda m: out.update(rating_min=float(m.group(1)))
    (take(r"\b(?:rated|rating(?:\s+of)?|with a rating)\s*(?:above|over|at least|of|>=|>|greater than)?\s*" + _NUM + r"(?![\d.])", setr)
     or take(r"\b(?:at least|above|over|>=)\s*" + _NUM + r"(?![\d.])(?:\s*stars?)?", setr)
     or take(r"\b" + _NUM + r"\s*\+(?:\s*stars?)?", setr)
     or take(r"\b" + _NUM + r"\s*stars?\b", setr))

    take(r"\b(?:starting|starts|beginning|begins)\s+with\s+(?:the\s+letter\s+)?([a-z0-9])\b",
         lambda m: out.update(letter=m.group(1)))
    take(r"\bby\s+([a-z][a-z .'\-]*?)(?=\s+(?:after|since|before|until|rated|published|starting|in\s+\d|between|from|with)\b|\s*$)",
         lambda m: out.update(author=m.group(1).strip()))

    out["rest"] = re.sub(r"\s+", " ", re.sub(_FILLER, " ", t)).strip(" .,;-")
    return out


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------
def _st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class AIBookRecommender:
    def __init__(self, books: pd.DataFrame, use_transformer: bool = True, cache_dir: str | None = None):
        df = books.reset_index(drop=True)
        for c, default in [("author", "Unknown"), ("genre", ""), ("description", ""),
                           ("rating", 0.0), ("ratings_count", 1), ("year", 0)]:
            if c not in df:
                df[c] = default
        df["author"] = df["author"].fillna("Unknown").astype(str)
        df["genre"] = df["genre"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["title"] = df["title"].astype(str)
        self.df = df
        self.n = len(df)

        self.title_l = df["title"].str.lower().str.strip()
        self.author_l = df["author"].str.lower()
        self.year = df["year"].fillna(0).astype(int).to_numpy()
        self.rating = df["rating"].fillna(0).astype(float).to_numpy()
        self.first = self.title_l.str[:1].to_numpy()

        self.tags = defaultdict(list)
        for i, g in enumerate(df["genre"]):
            for tag in {x.strip().lower() for x in g.split(",") if x.strip()}:
                self.tags[tag].append(i)

        # Embeddings: title. author. genre. genre. description
        cached = None
        if cache_dir and not (use_transformer and _st_available()):
            cached = load_index(cache_dir, df["title"])
        if cached is not None:  # fast start: reuse the saved index (no fitting, low memory)
            self.embedder, self.emb = cached
            self.model_name = "TF-IDF + SVD (LSA), pre-built index"
        else:
            texts = (df["title"] + ". " + df["author"] + ". " + df["genre"] + ". "
                     + df["genre"] + ". " + df["description"]).tolist()
            self.embedder = Embedder(use_transformer, dims=128 if self.n < 5000 else 256)  # small data needs fewer dims
            self.model_name = self.embedder.name
            self.emb = self.embedder.fit_transform(texts)
            del texts
        self.df = self.df.drop(columns=["description"])  # only needed to build embeddings
        df = self.df

        # Typo-tolerant title matching (character 3-grams)
        self.title_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True)
        self.title_mat = self.title_vec.fit_transform(self.title_l)

        # Bayesian weighted rating
        v = df["ratings_count"].fillna(1).astype(float).clip(lower=1).to_numpy()
        m = float(np.quantile(v, 0.6))
        C = float(self.rating[self.rating > 0].mean()) if (self.rating > 0).any() else 0.0
        wr = v / (v + m) * self.rating + m / (v + m) * C
        self.qual = (wr / 5.0).astype(np.float32)
        gc.collect()

    # ---- helpers -------------------------------------------------------
    def _rank(self, idx, qvec, k, sim_w=SIM_W):
        """Top-k indices from candidate idx (None = all books)."""
        full = idx is None
        if full:
            idx = np.arange(self.n)
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0 or k <= 0:
            return np.array([], dtype=int)
        if qvec is None:
            score = self.qual[idx]
        else:
            sims = self.emb @ qvec if full else self.emb[idx] @ qvec
            score = sim_w * sims + (1 - sim_w) * self.qual[idx]
        k = min(k, idx.size)
        part = np.argpartition(-score, k - 1)[:k] if k < idx.size else np.arange(idx.size)
        return idx[part[np.argsort(-score[part])]]

    def _frame(self, idx, why=None) -> pd.DataFrame:
        out = self.df.iloc[list(idx)][COLUMNS].copy()
        out["rating"] = out["rating"].astype(float).round(2)
        out["year"] = out["year"].fillna(0).astype(int)
        if why is not None:
            out["why"] = why
        return out.reset_index(drop=True)

    def _q(self, text):
        return self.embedder.encode([text])[0]

    def _empty(self, note):
        return pd.DataFrame(columns=COLUMNS), note

    def _find_title(self, title: str):
        """Return (index, how) for the best matching book, or (None, '')."""
        q = title.lower().strip()
        if not q or self.n == 0:
            return None, ""
        counts = self.df["ratings_count"].fillna(1).to_numpy()
        for how, mask in (("exact", self.title_l == q),
                          ("prefix", self.title_l.str.startswith(q)),
                          ("contains", self.title_l.str.contains(q, regex=False))):
            hit = np.flatnonzero(mask.to_numpy())
            if hit.size:
                return int(hit[np.argmax(counts[hit])]), how
        sc = (self.title_mat @ self.title_vec.transform([q]).T).toarray().ravel()
        top = np.argsort(-sc)[:40]
        if sc[top[0]] >= 0.30:
            # re-rank the closest-looking titles by spelling similarity and popularity
            import difflib
            pop = np.log10(np.maximum(counts[top], 1)) / max(np.log10(max(counts.max(), 10)), 1)
            ratio = np.array([difflib.SequenceMatcher(None, q, self.title_l.iloc[j][:len(q) + 2]).ratio()
                              for j in top])
            best = int(top[np.argmax(0.6 * ratio + 0.4 * pop)])
            return best, "fuzzy"
        return int((self.emb @ self._q(q)).argmax()), "semantic"

    # ---- request types -------------------------------------------------
    def by_genre(self, genre: str, k: int = 50):
        g = genre.lower().strip()
        if not g:
            return self._empty("Please enter a genre.")
        idx = self.tags.get(g)
        if idx is None:
            idx = [i for tag, lst in self.tags.items() if g in tag for i in lst] or None
        qv = self._q(f"{g}. {g}")
        if idx is None:
            res = self._rank(None, qv, k)
            return self._frame(res), f'No genre tag "{genre}" found; showing semantically similar books.'
        res = self._rank(idx, qv, k, sim_w=0.05)  # tag already matches: rank by rating + popularity, similarity only breaks ties
        return self._frame(res), f'{len(res)} books tagged "{genre}".'

    def by_author(self, author: str, k: int = 50):
        a = author.lower().strip()
        if not a:
            return self._empty("Please enter an author.")
        own = np.flatnonzero(self.author_l.str.contains(a, regex=False).to_numpy())
        if own.size == 0:
            res = self._rank(None, self._q(a), k)
            return self._frame(res), f'Author "{author}" not found; showing semantically related books.'
        own_top = self._rank(own, None, k)
        why = ["by this author"] * len(own_top)
        res = list(own_top)
        rest = k - len(res)
        if rest > 0:
            cen = normalize(self.emb[own].mean(axis=0, keepdims=True))[0]
            others = np.setdiff1d(np.arange(self.n), own)
            extra = self._rank(others, cen, rest)
            res += list(extra)
            why += ["readers of this author also like"] * len(extra)
        return self._frame(res, why), f'{len(own)} books by "{author}" plus similar authors.'

    def by_rating(self, lo: float = 0.0, hi: float = 5.0, k: int = 50):
        if lo > hi:
            lo, hi = hi, lo
        idx = np.flatnonzero((self.rating >= lo) & (self.rating <= hi))
        res = self._rank(idx, None, k)
        return self._frame(res), f"Books rated {lo:g} to {hi:g}, best quality first."

    def similar_to(self, title: str, k: int = 50, mask=None):
        i, how = self._find_title(title)
        if i is None:
            return self._empty("Please enter a book title.")
        cand = None if mask is None else np.flatnonzero(mask)
        if cand is not None:
            cand = cand[cand != i]
        else:
            cand = np.flatnonzero(np.arange(self.n) != i)
        res = self._rank(cand, self.emb[i], k)
        row = self.df.iloc[i]
        note = f'Matched: "{row.title}" by {row.author}' + ("" if how == "exact" else f" ({how} match)")
        return self._frame(res), note

    def by_first_letter(self, letter: str, k: int = 50):
        ch = letter.lower().strip()[:1]
        if not ch:
            return self._empty("Please enter a letter.")
        res = self._rank(np.flatnonzero(self.first == ch), None, k)
        return self._frame(res), f'Titles starting with "{ch.upper()}".'

    def by_year(self, lo: int = 0, hi: int = 9999, k: int = 50):
        if lo > hi:
            lo, hi = hi, lo
        idx = np.flatnonzero((self.year > 0) & (self.year >= lo) & (self.year <= hi))
        res = self._rank(idx, None, k)
        label = f"{lo}" if lo == hi else f"{lo or 'earliest'} to {hi if hi < 9999 else 'latest'}"
        return self._frame(res), f"Books published {label}."

    def search(self, text: str, k: int = 50):
        if not text.strip():
            return self._empty("Please type a request.")
        p = parse_query(text)
        mask = np.ones(self.n, dtype=bool)
        parts = []
        if p["year_lo"] is not None or p["year_hi"] is not None:
            lo = p["year_lo"] if p["year_lo"] is not None else 0
            hi = p["year_hi"] if p["year_hi"] is not None else 9999
            mask &= (self.year > 0) & (self.year >= lo) & (self.year <= hi)
            parts.append(f"year {lo if lo else '...'}-{hi if hi < 9999 else '...'}")
        if p["rating_min"] is not None:
            mask &= self.rating >= p["rating_min"]
            parts.append(f"rating >= {p['rating_min']:g}")
        if p["letter"]:
            mask &= self.first == p["letter"]
            parts.append(f'title starts with "{p["letter"].upper()}"')
        if p["author"]:
            mask &= self.author_l.str.contains(p["author"], regex=False).to_numpy()
            parts.append(f'author contains "{p["author"]}"')

        if p["like"]:
            df, note = self.similar_to(p["like"], k, mask=mask)
        else:
            idx = np.flatnonzero(mask)
            qv = self._q(p["rest"]) if p["rest"] else None
            res = self._rank(idx, qv, k)
            df, note = self._frame(res), ""
        if parts:
            note = (note + " | " if note else "") + "Filters: " + ", ".join(parts)
        if p["rest"] and not p["like"]:
            note = (note + " | " if note else "") + f'Topic: "{p["rest"]}"'
        return df, note

    def for_reader(self, liked_titles: list[str], k: int = 20):
        """Recommend from the average of several liked books (no web tab in v1.0)."""
        ids = [i for i, _ in (self._find_title(t) for t in liked_titles) if i is not None]
        if not ids:
            return self._empty("None of those titles were found.")
        cen = normalize(self.emb[ids].mean(axis=0, keepdims=True))[0]
        others = np.setdiff1d(np.arange(self.n), ids)
        return self._frame(self._rank(others, cen, k)), f"Based on {len(ids)} liked book(s)."


# --------------------------------------------------------------------------
# Test / demo data
# --------------------------------------------------------------------------
_G = {
    "Fantasy": ("dragon wizard kingdom sword quest magic elves prophecy throne realm sorcerer",
                ["Dragon", "Crown", "Sword", "Wizard", "Throne", "Realm", "Spell", "Elf"]),
    "Science Fiction": ("spaceship galaxy robot alien future colony planet android starship empire",
                        ["Galaxy", "Orbit", "Android", "Colony", "Signal", "Nebula", "Starship", "Void"]),
    "Mystery": ("detective murder clue suspect investigation crime inspector alibi secret case",
                ["Murder", "Clue", "Alibi", "Case", "Secret", "Shadow", "Witness", "Verdict"]),
    "Romance": ("love heart wedding passion kiss relationship couple affair longing devotion",
                ["Heart", "Kiss", "Promise", "Summer", "Letters", "Vow", "Embrace", "Dance"]),
    "Horror": ("ghost haunted monster blood terror nightmare curse demon dark graveyard",
               ["Haunting", "Nightmare", "Curse", "Graveyard", "Whisper", "Blood", "Demon", "Dark"]),
    "Historical Fiction": ("war empire revolution queen battle century soldier court victorian migration",
                           ["Empire", "Revolution", "Queen", "Siege", "Frontier", "Winter", "Banner", "Harbor"]),
    "Thriller": ("conspiracy spy chase assassin agent heist hostage danger ticking escape",
                 ["Protocol", "Conspiracy", "Pursuit", "Agent", "Deadline", "Target", "Cipher", "Hunter"]),
    "Biography": ("life memoir journey childhood career struggle legacy truth years remarkable",
                  ["Life", "Memoir", "Years", "Legacy", "Journey", "Story", "Voice", "Road"]),
}
_ADJ = ["Silent", "Lost", "Golden", "Broken", "Hidden", "Last", "Crimson", "Forgotten", "Endless",
        "Burning", "Frozen", "Wandering", "Distant", "Final", "Secret", "Ancient", "Quiet", "Wild"]
_PLACE = ["the North", "Avalon", "Marrow Bay", "the Deep", "Eldmoor", "Kestrel Hill", "the Mist",
          "Ravenhold", "Lowmere", "the Ashlands", "Starfall", "Greywater", "Ironvale", "Thornwick"]
_FIRST = ["Alan", "Maria", "John", "Priya", "Elena", "David", "Sofia", "Omar", "Grace", "Liam",
          "Hana", "Victor", "Nora", "Arjun", "Clara", "Felix", "Isla", "Marcus", "Tara", "Henry"]
_LAST = ["Hart", "Nair", "Okafor", "Lindqvist", "Moreau", "Bianchi", "Kowalski", "Tanaka", "Fischer",
         "Reyes", "Abbott", "Verma", "Dubois", "Novak", "Sinclair", "Quinn", "Ashby", "Rahman",
         "Whitlock", "Ferreira", "Gallo", "Hendricks", "Ito", "Jansen", "Kaur", "Lund", "Mendes"]


def synthetic_books(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate n random but genre-coherent books for testing."""
    rng = np.random.default_rng(seed)
    genres = list(_G)
    authors = [f"{f} {l}" for f in _FIRST for l in _LAST]
    # each author sticks mostly to one genre
    author_genre = {a: genres[rng.integers(len(genres))] for a in authors}
    rows, seen = [], set()
    while len(rows) < n:
        a = authors[rng.integers(len(authors))]
        g1 = author_genre[a] if rng.random() < 0.8 else genres[rng.integers(len(genres))]
        words, nouns = _G[g1]
        title = f"{_ADJ[rng.integers(len(_ADJ))]} {nouns[rng.integers(len(nouns))]} of {_PLACE[rng.integers(len(_PLACE))]}"
        if (title, a) in seen:
            title += f" {int(rng.integers(2, 99))}"
            if (title, a) in seen:
                continue
        seen.add((title, a))
        gl = [g1]
        if rng.random() < 0.5:
            gl.append(genres[rng.integers(len(genres))])
        if rng.random() < 0.3:
            gl.append("Fiction")
        w = _G[g1][0].split()
        desc = " ".join(rng.choice(w, size=int(rng.integers(12, 25))))
        cnt = int(np.exp(rng.normal(6.5, 2.0)))
        rows.append({
            "title": title, "author": a, "genre": ", ".join(dict.fromkeys(gl)),
            "rating": float(np.clip(rng.normal(3.9, 0.45), 1.0, 5.0)),
            "ratings_count": max(1, cnt), "year": int(rng.integers(1950, 2024)),
            "description": desc,
        })
    return pd.DataFrame(rows)


def demo_data() -> pd.DataFrame:
    """A small hand-written catalogue for the text interface."""
    d = [
        ("Dune", "Frank Herbert", "Science Fiction, Fantasy", 4.27, 1200000, 1965, "desert planet spice politics empire prophecy Paul Atreides"),
        ("The Hobbit", "J.R.R. Tolkien", "Fantasy, Classics", 4.28, 3500000, 1937, "hobbit dragon quest dwarves treasure wizard adventure"),
        ("The Fellowship of the Ring", "J.R.R. Tolkien", "Fantasy, Classics", 4.38, 2500000, 1954, "ring dark lord quest fellowship middle earth wizard"),
        ("The Name of the Wind", "Patrick Rothfuss", "Fantasy", 4.52, 750000, 2007, "magic university musician legend tale wizard"),
        ("A Game of Thrones", "George R.R. Martin", "Fantasy", 4.45, 2100000, 1996, "throne kingdom war houses dragons intrigue"),
        ("Neuromancer", "William Gibson", "Science Fiction", 3.89, 400000, 1984, "hacker cyberspace artificial intelligence future corporation"),
        ("Foundation", "Isaac Asimov", "Science Fiction", 4.17, 600000, 1951, "galactic empire psychohistory future collapse"),
        ("Ender's Game", "Orson Scott Card", "Science Fiction", 4.30, 1300000, 1985, "children military school alien war strategy"),
        ("The Martian", "Andy Weir", "Science Fiction", 4.42, 1000000, 2011, "astronaut mars survival science stranded"),
        ("Gone Girl", "Gillian Flynn", "Thriller, Mystery", 4.12, 2000000, 2012, "marriage disappearance suspect twist crime"),
        ("The Girl with the Dragon Tattoo", "Stieg Larsson", "Mystery, Thriller", 4.11, 2500000, 2005, "journalist hacker investigation murder conspiracy"),
        ("And Then There Were None", "Agatha Christie", "Mystery, Classics", 4.27, 1500000, 1939, "island murder guests detective suspect"),
        ("Murder on the Orient Express", "Agatha Christie", "Mystery, Classics", 4.19, 1100000, 1934, "train detective Poirot murder suspects"),
        ("Pride and Prejudice", "Jane Austen", "Romance, Classics", 4.29, 4000000, 1813, "love marriage society class family"),
        ("Outlander", "Diana Gabaldon", "Romance, Historical Fiction", 4.25, 900000, 1991, "time travel nurse Scotland love war"),
        ("The Notebook", "Nicholas Sparks", "Romance", 4.11, 1000000, 1996, "love letters summer couple memory"),
        ("The Shining", "Stephen King", "Horror", 4.26, 1100000, 1977, "haunted hotel winter family madness ghost"),
        ("It", "Stephen King", "Horror", 4.25, 1000000, 1986, "clown monster town childhood terror friends"),
        ("Dracula", "Bram Stoker", "Horror, Classics", 4.01, 900000, 1897, "vampire count castle blood terror"),
        ("The Hunger Games", "Suzanne Collins", "Science Fiction, Young Adult", 4.34, 7000000, 2008, "arena survival rebellion dystopia tribute"),
        ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling", "Fantasy, Young Adult", 4.47, 9000000, 1997, "wizard school magic friends boy orphan"),
        ("The Da Vinci Code", "Dan Brown", "Thriller, Mystery", 3.85, 2000000, 2003, "symbologist church secret conspiracy chase puzzle"),
        ("The Book Thief", "Markus Zusak", "Historical Fiction", 4.39, 2200000, 2005, "war Germany girl books death narrator"),
        ("Steve Jobs", "Walter Isaacson", "Biography", 4.15, 500000, 2011, "technology life career company innovator"),
        ("Educated", "Tara Westover", "Biography, Memoir", 4.47, 1100000, 2018, "memoir family education survivalist childhood"),
        ("Gideon the Ninth", "Tamsyn Muir", "Fantasy, Science Fiction", 4.10, 150000, 2019, "necromancer swordswoman haunted space palace"),
    ]
    return pd.DataFrame(d, columns=["title", "author", "genre", "rating", "ratings_count", "year", "description"])


# --------------------------------------------------------------------------
# Text interface
# --------------------------------------------------------------------------
def _show(df, note):
    if note:
        print(note)
    if df.empty:
        print("No books found. Try different input.")
        return
    print(df.to_string(index=False, max_colwidth=45))


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "books.db"
    if os.path.exists(db):
        import book_db
        books = book_db.load_books(db)
    else:
        print(f"{db} not found; using the small demo catalogue.")
        books = demo_data()
    rec = AIBookRecommender(books)
    print(f"{len(books)} books | model: {rec.model_name}")
    print("Type a request, or: genre X | author X | rating LO HI | similar TITLE | letter L | year LO [HI] | quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("quit", "exit", "q"):
            break
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "genre":
                _show(*rec.by_genre(arg, 20))
            elif cmd == "author":
                _show(*rec.by_author(arg, 20))
            elif cmd == "rating":
                v = [float(x) for x in arg.split()]
                _show(*rec.by_rating(v[0], v[1] if len(v) > 1 else 5.0, 20))
            elif cmd == "similar":
                _show(*rec.similar_to(arg, 20))
            elif cmd == "letter":
                _show(*rec.by_first_letter(arg, 20))
            elif cmd == "year":
                v = [int(x) for x in arg.split()]
                _show(*rec.by_year(v[0], v[1] if len(v) > 1 else v[0], 20))
            else:
                _show(*rec.search(line, 20))
        except (ValueError, IndexError):
            print("Please enter valid numbers.")


if __name__ == "__main__":
    main()
