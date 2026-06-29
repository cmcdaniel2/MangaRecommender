from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_TRAINING_PATH = Path("data/processed/manga_training.csv")
DEFAULT_MODEL_DIR = Path("data/models/sentence_transformer_baseline")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def load_sentence_model(
    model_name: str,
    device: str | None,
    max_seq_length: int | None,
) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    return model


def build_embedding_baseline(
    training_path: Path,
    model_dir: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    max_seq_length: int | None,
) -> dict[str, Any]:
    df = load_training_data(training_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    model = load_sentence_model(model_name, device=device, max_seq_length=max_seq_length)
    texts = df["model_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    metadata = df[[column for column in METADATA_COLUMNS if column in df.columns]].copy()
    metadata_path = model_dir / "manga_metadata.csv"
    embeddings_path = model_dir / "manga_embeddings.npy"
    config_path = model_dir / "embedding_config.json"

    metadata.to_csv(metadata_path, index=False)
    np.save(embeddings_path, embeddings)

    config: dict[str, Any] = {
        "training_path": str(training_path),
        "metadata_path": str(metadata_path),
        "embeddings_path": str(embeddings_path),
        "model_name": model_name,
        "row_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "device": str(model.device),
        "max_seq_length": int(model.max_seq_length),
        "normalize_embeddings": True,
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    return config


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "embedding_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Embedding config not found in {model_dir}. Run the build command first."
        )

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_artifacts(model_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    config = load_config(model_dir)
    metadata_path = Path(config["metadata_path"])
    embeddings_path = Path(config["embeddings_path"])

    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding artifacts not found in {model_dir}. Run the build command first."
        )

    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    embeddings = np.load(embeddings_path).astype("float32")

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match metadata rows ({len(metadata)})."
        )

    return metadata, embeddings, config


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


def top_k_from_scores(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
    query_title: str,
    query_mangadex_id: str,
    excluded_index: int | None = None,
) -> pd.DataFrame:
    if excluded_index is not None:
        scores = scores.copy()
        scores[excluded_index] = -1.0

    candidate_count = min(top_k, len(scores) - (1 if excluded_index is not None else 0))
    if candidate_count <= 0:
        return pd.DataFrame()

    top_indices = np.argsort(scores)[::-1][:candidate_count]
    rows = metadata.iloc[top_indices].copy()
    rows.insert(0, "rank", range(1, len(rows) + 1))
    rows.insert(1, "cosine_similarity", [round(float(scores[index]), 4) for index in top_indices])
    rows.insert(2, "query_title", query_title)
    rows.insert(3, "query_mangadex_id", query_mangadex_id)

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


def recommend_similar(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    query: str,
    top_k: int,
) -> pd.DataFrame:
    query_index = find_manga_index(metadata, query)
    query_vector = embeddings[query_index]
    scores = embeddings @ query_vector

    return top_k_from_scores(
        metadata=metadata,
        scores=scores,
        top_k=top_k,
        query_title=metadata.iloc[query_index]["title"],
        query_mangadex_id=metadata.iloc[query_index]["mangadex_id"],
        excluded_index=query_index,
    )


def search_by_text(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    config: dict[str, Any],
    query_text: str,
    top_k: int,
    device: str | None,
) -> pd.DataFrame:
    model = load_sentence_model(
        config["model_name"],
        device=device,
        max_seq_length=int(config["max_seq_length"]),
    )
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")[0]
    scores = embeddings @ query_embedding

    return top_k_from_scores(
        metadata=metadata,
        scores=scores,
        top_k=top_k,
        query_title=query_text,
        query_mangadex_id="free_text_query",
    )


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
    parser = argparse.ArgumentParser(
        description="Sentence-transformer cosine-similarity manga recommender."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build dense embedding artifacts.")
    build_parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    build_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    build_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--device", default=None, help="Example: cpu, cuda, mps. Defaults to auto.")
    build_parser.add_argument("--max-seq-length", type=int, default=256)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend similar manga.")
    recommend_parser.add_argument("query", help="MangaDex ID, exact title, partial title, or alt title.")
    recommend_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    recommend_parser.add_argument("--top-k", type=int, default=10)
    recommend_parser.add_argument("--output", type=Path, default=None)

    search_parser = subparsers.add_parser("search", help="Search manga by arbitrary text.")
    search_parser.add_argument("query_text", help="Free text description to embed and search with.")
    search_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--device", default=None, help="Example: cpu, cuda, mps. Defaults to auto.")
    search_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        config = build_embedding_baseline(
            training_path=args.training_path,
            model_dir=args.model_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
        print(f"Built sentence-transformer baseline in: {args.model_dir}")
        print(f"Model: {config['model_name']}")
        print(f"Rows: {config['row_count']}")
        print(f"Embedding dimension: {config['embedding_dimension']}")
        print(f"Device: {config['device']}")
        return

    if args.command == "recommend":
        metadata, embeddings, _config = load_model_artifacts(args.model_dir)
        result = recommend_similar(metadata, embeddings, args.query, args.top_k)
        write_recommendations(result, args.output)
        return

    if args.command == "search":
        metadata, embeddings, config = load_model_artifacts(args.model_dir)
        result = search_by_text(
            metadata=metadata,
            embeddings=embeddings,
            config=config,
            query_text=args.query_text,
            top_k=args.top_k,
            device=args.device,
        )
        write_recommendations(result, args.output)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
