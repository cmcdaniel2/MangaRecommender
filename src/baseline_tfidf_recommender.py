from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_TRAINING_PATH = Path("data/processed/manga_training.csv")
DEFAULT_MODEL_DIR = Path("data/models/tfidf_baseline")

METADATA_COLUMNS = [
    "mangadex_id",
    "title",
    "alt_titles_latin",
    "description",
    "original_language",
    "publication_demographic",
    "status",
    "year",
    "content_rating",
    "tags",
    "genres",
    "themes",
    "formats",
    "authors",
    "artists",
    "mangadex_url",
]


def normalize_lookup_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def split_pipe_list(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run src/process_mangadex.py first."
        )

    df = pd.read_csv(path, dtype=str).fillna("")
    required_columns = {"mangadex_id", "title", "model_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")

    df = df.drop_duplicates(subset=["mangadex_id"]).reset_index(drop=True)
    df["model_text"] = df["model_text"].astype(str)
    df = df[df["model_text"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No non-empty model_text rows found.")

    return df


def build_tfidf_baseline(
    training_path: Path,
    model_dir: Path,
    max_features: int | None,
    min_df: int,
    max_df: float,
) -> dict[str, Any]:
    df = load_training_data(training_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[\w'-]{2,}\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(df["model_text"])

    metadata = df[[column for column in METADATA_COLUMNS if column in df.columns]].copy()
    metadata_path = model_dir / "manga_metadata.csv"
    matrix_path = model_dir / "tfidf_matrix.npz"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    config_path = model_dir / "baseline_config.json"

    metadata.to_csv(metadata_path, index=False)
    sparse.save_npz(matrix_path, tfidf_matrix)
    joblib.dump(vectorizer, vectorizer_path)

    config: dict[str, Any] = {
        "training_path": str(training_path),
        "metadata_path": str(metadata_path),
        "matrix_path": str(matrix_path),
        "vectorizer_path": str(vectorizer_path),
        "row_count": int(tfidf_matrix.shape[0]),
        "feature_count": int(tfidf_matrix.shape[1]),
        "vectorizer": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "sublinear_tf": True,
            "norm": "l2",
        },
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    return config


def load_model(model_dir: Path) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    metadata_path = model_dir / "manga_metadata.csv"
    matrix_path = model_dir / "tfidf_matrix.npz"

    if not metadata_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. Run the build command first."
        )

    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    matrix = sparse.load_npz(matrix_path).tocsr()

    if matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Matrix row count ({matrix.shape[0]}) does not match metadata rows ({len(metadata)})."
        )

    return metadata, matrix


def find_manga_index(metadata: pd.DataFrame, query: str) -> int:
    normalized_query = normalize_lookup_text(query)
    if not normalized_query:
        raise ValueError("Please provide a non-empty MangaDex ID or title.")

    id_matches = metadata.index[
        metadata["mangadex_id"].astype(str).map(normalize_lookup_text) == normalized_query
    ].tolist()
    if id_matches:
        return id_matches[0]

    title_matches = metadata.index[
        metadata["title"].astype(str).map(normalize_lookup_text) == normalized_query
    ].tolist()
    if title_matches:
        return title_matches[0]

    contains_matches = metadata.index[
        metadata["title"].astype(str).map(normalize_lookup_text).str.contains(
            re.escape(normalized_query), regex=True
        )
    ].tolist()
    if contains_matches:
        return contains_matches[0]

    alt_matches = []
    for index, alt_titles in metadata["alt_titles_latin"].items():
        candidates = [normalize_lookup_text(title) for title in split_pipe_list(alt_titles)]
        if normalized_query in candidates or any(normalized_query in title for title in candidates):
            alt_matches.append(index)

    if alt_matches:
        return alt_matches[0]

    raise ValueError(f"No manga found for query: {query!r}")


def recommend_similar(
    metadata: pd.DataFrame,
    matrix: sparse.csr_matrix,
    query: str,
    top_k: int,
) -> pd.DataFrame:
    query_index = find_manga_index(metadata, query)
    query_vector = matrix[query_index]

    # TF-IDF rows are L2-normalized, so the dot product is cosine similarity.
    scores = (matrix @ query_vector.T).toarray().ravel()
    scores[query_index] = -1.0

    candidate_count = min(top_k, len(scores) - 1)
    if candidate_count <= 0:
        return pd.DataFrame()

    top_indices = scores.argsort()[::-1][:candidate_count]

    rows = metadata.iloc[top_indices].copy()
    rows.insert(0, "rank", range(1, len(rows) + 1))
    rows.insert(1, "cosine_similarity", [round(float(scores[index]), 4) for index in top_indices])
    rows.insert(2, "query_title", metadata.iloc[query_index]["title"])
    rows.insert(3, "query_mangadex_id", metadata.iloc[query_index]["mangadex_id"])

    output_columns = [
        "rank",
        "cosine_similarity",
        "query_title",
        "query_mangadex_id",
        "title",
        "mangadex_id",
        "genres",
        "themes",
        "publication_demographic",
        "status",
        "year",
        "mangadex_url",
    ]
    return rows[[column for column in output_columns if column in rows.columns]]


def write_recommendations(result: pd.DataFrame, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved recommendations to: {output_path}")
        return

    if result.empty:
        print("No recommendations found.")
        return

    display = result.copy()
    for column in ["genres", "themes"]:
        if column in display.columns:
            display[column] = display[column].astype(str).str.slice(0, 90)

    print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF cosine-similarity manga recommender.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build TF-IDF baseline artifacts.")
    build_parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    build_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    build_parser.add_argument("--max-features", type=int, default=50000)
    build_parser.add_argument("--min-df", type=int, default=2)
    build_parser.add_argument("--max-df", type=float, default=0.85)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend similar manga.")
    recommend_parser.add_argument("query", help="MangaDex ID, exact title, partial title, or alt title.")
    recommend_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    recommend_parser.add_argument("--top-k", type=int, default=10)
    recommend_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        max_features = args.max_features if args.max_features > 0 else None
        config = build_tfidf_baseline(
            training_path=args.training_path,
            model_dir=args.model_dir,
            max_features=max_features,
            min_df=args.min_df,
            max_df=args.max_df,
        )
        print(f"Built TF-IDF baseline in: {args.model_dir}")
        print(f"Rows: {config['row_count']}")
        print(f"Features: {config['feature_count']}")
        return

    if args.command == "recommend":
        metadata, matrix = load_model(args.model_dir)
        result = recommend_similar(metadata, matrix, args.query, args.top_k)
        write_recommendations(result, args.output)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
