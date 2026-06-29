from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DEFAULT_RAW_CSV = RAW_DIR / "mangadex_manga_raw.csv"
DEFAULT_RAW_JSONL = RAW_DIR / "mangadex_manga_raw.jsonl"
DEFAULT_TAGS_JSON = RAW_DIR / "mangadex_tags.json"


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SEPARATOR_RE = re.compile(r"\n\s*-{3,}\s*\n")
MULTISPACE_RE = re.compile(r"\s+")
NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


TRAINING_FIELDS = [
    "mangadex_id",
    "title",
    "alt_titles_latin",
    "description",
    "model_text",
    "original_language",
    "publication_demographic",
    "status",
    "year",
    "content_rating",
    "last_volume",
    "last_chapter",
    "tags",
    "genres",
    "themes",
    "formats",
    "content_tags",
    "unknown_tags",
    "authors",
    "artists",
    "cover_file",
    "mangadex_url",
    "description_word_count",
    "model_text_word_count",
]


PAIR_FIELDS = [
    "manga_id_a",
    "title_a",
    "manga_id_b",
    "title_b",
    "label",
    "source",
    "relationship_types",
    "shared_tags",
    "tag_jaccard",
]


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "").strip())


def clean_description(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    sections = SEPARATOR_RE.split(text, maxsplit=1)
    if sections and sections[0].strip():
        text = sections[0]

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower().strip("*:_ ")

        if not stripped:
            kept_lines.append(" ")
            continue

        if lowered in {"links", "official links", "external links"}:
            continue

        if "http://" in lowered or "https://" in lowered or stripped.startswith("- ["):
            continue

        kept_lines.append(stripped)

    text = "\n".join(kept_lines)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\[/?(?:spoiler|b|i|u|url)[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    text = text.replace(">", " ")

    return normalize_whitespace(text)


def split_csv_list(value: str) -> list[str]:
    return sorted({normalize_whitespace(part) for part in (value or "").split(",") if part.strip()})


def split_pipe_list(value: str) -> list[str]:
    return sorted({normalize_whitespace(part) for part in (value or "").split("|") if part.strip()})


def looks_latinish(text: str) -> bool:
    text = normalize_whitespace(text)
    if not text:
        return False

    ascii_alnum = sum(1 for char in text if char.isascii() and char.isalnum())
    non_ascii = sum(1 for char in text if not char.isascii())

    return ascii_alnum > 0 and non_ascii <= max(4, int(len(text) * 0.35))


def filtered_alt_titles(value: str, primary_title: str, limit: int = 10) -> list[str]:
    primary_key = normalize_whitespace(primary_title).casefold()
    titles = []

    for title in split_pipe_list(value):
        key = title.casefold()
        if key == primary_key:
            continue
        if len(title) > 100:
            continue
        if looks_latinish(title):
            titles.append(title)

    return sorted(set(titles), key=lambda item: (len(item), item.casefold()))[:limit]


def clean_year(value: str) -> str:
    value = normalize_whitespace(value)
    if not value:
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def slugify(value: str) -> str:
    slug = NON_SLUG_RE.sub("_", value.lower()).strip("_")
    return slug or "unknown"


def pipe_join(values: list[str] | set[str]) -> str:
    return " | ".join(sorted(values))


def word_count(value: str) -> int:
    return len([token for token in normalize_whitespace(value).split(" ") if token])


def load_tag_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    tag_groups: dict[str, str] = {}
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        name = (attrs.get("name") or {}).get("en")
        group = attrs.get("group")
        if name and group:
            tag_groups[name.casefold()] = group

    return tag_groups


def load_raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_valid_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    if not path.exists():
        return valid_rows, invalid_rows

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                valid_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                invalid_rows.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "line_length": len(line),
                    }
                )

    return valid_rows, invalid_rows


def build_model_text(row: dict[str, Any]) -> str:
    parts = [f"Title: {row['title']}."]

    if row["alt_titles_latin"]:
        parts.append(f"Alternate titles: {row['alt_titles_latin']}.")
    if row["description"]:
        parts.append(f"Description: {row['description']}")
    if row["genres"]:
        parts.append(f"Genres: {row['genres']}.")
    if row["themes"]:
        parts.append(f"Themes: {row['themes']}.")
    if row["formats"]:
        parts.append(f"Formats: {row['formats']}.")
    if row["publication_demographic"]:
        parts.append(f"Demographic: {row['publication_demographic']}.")
    if row["authors"]:
        parts.append(f"Authors: {row['authors']}.")
    if row["artists"]:
        parts.append(f"Artists: {row['artists']}.")

    return normalize_whitespace(" ".join(parts))


def process_rows(raw_rows: list[dict[str, str]], tag_groups: dict[str, str]) -> list[dict[str, Any]]:
    processed_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in raw_rows:
        manga_id = normalize_whitespace(raw.get("mangadex_id", ""))
        if not manga_id or manga_id in seen_ids:
            continue
        seen_ids.add(manga_id)

        title = normalize_whitespace(raw.get("title", ""))
        tags = split_csv_list(raw.get("tags", ""))

        grouped_tags: dict[str, list[str]] = {
            "genre": [],
            "theme": [],
            "format": [],
            "content": [],
            "unknown": [],
        }
        for tag in tags:
            group = tag_groups.get(tag.casefold(), "unknown")
            grouped_tags.setdefault(group, []).append(tag)

        row: dict[str, Any] = {
            "mangadex_id": manga_id,
            "title": title,
            "alt_titles_latin": pipe_join(filtered_alt_titles(raw.get("alt_titles", ""), title)),
            "description": clean_description(raw.get("description", "")),
            "original_language": normalize_whitespace(raw.get("original_language", "")),
            "publication_demographic": normalize_whitespace(raw.get("publication_demographic", "")),
            "status": normalize_whitespace(raw.get("status", "")),
            "year": clean_year(raw.get("year", "")),
            "content_rating": normalize_whitespace(raw.get("content_rating", "")),
            "last_volume": normalize_whitespace(raw.get("last_volume", "")),
            "last_chapter": normalize_whitespace(raw.get("last_chapter", "")),
            "tags": pipe_join(tags),
            "genres": pipe_join(grouped_tags.get("genre", [])),
            "themes": pipe_join(grouped_tags.get("theme", [])),
            "formats": pipe_join(grouped_tags.get("format", [])),
            "content_tags": pipe_join(grouped_tags.get("content", [])),
            "unknown_tags": pipe_join(grouped_tags.get("unknown", [])),
            "authors": pipe_join(split_pipe_list(raw.get("authors", ""))),
            "artists": pipe_join(split_pipe_list(raw.get("artists", ""))),
            "cover_file": normalize_whitespace(raw.get("cover_file", "")),
            "mangadex_url": f"https://mangadex.org/title/{manga_id}",
        }
        row["model_text"] = build_model_text(row)
        row["description_word_count"] = str(word_count(row["description"]))
        row["model_text_word_count"] = str(word_count(row["model_text"]))
        processed_rows.append(row)

    return processed_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    list_fields = {
        "alt_titles_latin",
        "tags",
        "genres",
        "themes",
        "formats",
        "content_tags",
        "unknown_tags",
        "authors",
        "artists",
    }

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            payload: dict[str, Any] = dict(row)
            for field in list_fields:
                payload[field] = split_pipe_list(str(payload.get(field, "")))
            payload["description_word_count"] = int(payload["description_word_count"])
            payload["model_text_word_count"] = int(payload["model_text_word_count"])
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_tag_matrix(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    observed_tags = sorted({tag for row in rows for tag in split_pipe_list(row["tags"])})
    tag_columns = [f"tag_{slugify(tag)}" for tag in observed_tags]

    matrix_rows = []
    for row in rows:
        row_tags = set(split_pipe_list(row["tags"]))
        matrix_row: dict[str, Any] = {
            "mangadex_id": row["mangadex_id"],
            "title": row["title"],
        }
        for tag, column in zip(observed_tags, tag_columns):
            matrix_row[column] = 1 if tag in row_tags else 0
        matrix_rows.append(matrix_row)

    return matrix_rows, ["mangadex_id", "title", *tag_columns]


def tag_jaccard(tags_a: set[str], tags_b: set[str]) -> float:
    if not tags_a and not tags_b:
        return 0.0
    union = tags_a | tags_b
    if not union:
        return 0.0
    return len(tags_a & tags_b) / len(union)


def build_relationship_pairs(
    json_rows: list[dict[str, Any]],
    processed_rows: list[dict[str, Any]],
    negative_ratio: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_id = {row["mangadex_id"]: row for row in processed_rows}
    tag_sets = {row["mangadex_id"]: set(split_pipe_list(row["tags"])) for row in processed_rows}
    relationship_types: dict[tuple[str, str], set[str]] = defaultdict(set)

    for json_row in json_rows:
        manga_id = json_row.get("id")
        if manga_id not in by_id:
            continue

        for relationship in json_row.get("relationships", []):
            if relationship.get("type") != "manga":
                continue

            related_id = relationship.get("id")
            if not related_id or related_id == manga_id or related_id not in by_id:
                continue

            pair = tuple(sorted((manga_id, related_id)))
            relationship_types[pair].add(relationship.get("related") or "unknown")

    pair_rows: list[dict[str, Any]] = []
    for manga_id_a, manga_id_b in sorted(relationship_types):
        pair_rows.append(
            make_pair_row(
                by_id,
                tag_sets,
                manga_id_a,
                manga_id_b,
                label=1,
                source="mangadex_relationship",
                relationship_types=relationship_types[(manga_id_a, manga_id_b)],
            )
        )

    positive_pairs = set(relationship_types)
    negative_count = len(pair_rows) * max(0, negative_ratio)
    pair_rows.extend(sample_negative_pairs(by_id, tag_sets, positive_pairs, negative_count, seed))

    return pair_rows


def make_pair_row(
    by_id: dict[str, dict[str, Any]],
    tag_sets: dict[str, set[str]],
    manga_id_a: str,
    manga_id_b: str,
    label: int,
    source: str,
    relationship_types: set[str] | None = None,
) -> dict[str, Any]:
    tags_a = tag_sets[manga_id_a]
    tags_b = tag_sets[manga_id_b]

    return {
        "manga_id_a": manga_id_a,
        "title_a": by_id[manga_id_a]["title"],
        "manga_id_b": manga_id_b,
        "title_b": by_id[manga_id_b]["title"],
        "label": label,
        "source": source,
        "relationship_types": pipe_join(relationship_types or set()),
        "shared_tags": len(tags_a & tags_b),
        "tag_jaccard": f"{tag_jaccard(tags_a, tags_b):.4f}",
    }


def sample_negative_pairs(
    by_id: dict[str, dict[str, Any]],
    tag_sets: dict[str, set[str]],
    positive_pairs: set[tuple[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    manga_ids = sorted(by_id)
    sampled_pairs: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(1000, count * 200)

    while len(sampled_pairs) < count and attempts < max_attempts:
        attempts += 1
        manga_id_a, manga_id_b = rng.sample(manga_ids, 2)
        pair = tuple(sorted((manga_id_a, manga_id_b)))

        if pair in positive_pairs or pair in sampled_pairs:
            continue
        if tag_sets[pair[0]] & tag_sets[pair[1]]:
            continue

        sampled_pairs.add(pair)

    while len(sampled_pairs) < count and attempts < max_attempts * 2:
        attempts += 1
        manga_id_a, manga_id_b = rng.sample(manga_ids, 2)
        pair = tuple(sorted((manga_id_a, manga_id_b)))

        if pair in positive_pairs or pair in sampled_pairs:
            continue
        if tag_jaccard(tag_sets[pair[0]], tag_sets[pair[1]]) > 0.1:
            continue

        sampled_pairs.add(pair)

    return [
        make_pair_row(
            by_id,
            tag_sets,
            manga_id_a,
            manga_id_b,
            label=0,
            source="random_no_or_low_tag_overlap",
        )
        for manga_id_a, manga_id_b in sorted(sampled_pairs)
    ]


def write_report(
    path: Path,
    raw_rows: list[dict[str, str]],
    processed_rows: list[dict[str, Any]],
    json_rows: list[dict[str, Any]],
    invalid_jsonl_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> None:
    positive_pairs = sum(1 for row in pair_rows if row["label"] == 1)
    negative_pairs = sum(1 for row in pair_rows if row["label"] == 0)
    empty_descriptions = sum(1 for row in processed_rows if not row["description"])
    observed_tags = sorted({tag for row in processed_rows for tag in split_pipe_list(row["tags"])})

    report = {
        "raw_csv_rows": len(raw_rows),
        "processed_rows": len(processed_rows),
        "valid_jsonl_rows": len(json_rows),
        "invalid_jsonl_rows": invalid_jsonl_rows,
        "empty_clean_descriptions": empty_descriptions,
        "observed_tag_count": len(observed_tags),
        "weak_positive_pairs": positive_pairs,
        "weak_negative_pairs": negative_pairs,
        "outputs": output_paths,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MangaDex raw exports into training data.")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--tags-json", type=Path, default=DEFAULT_TAGS_JSON)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_csv(args.raw_csv)
    tag_groups = load_tag_groups(args.tags_json)
    json_rows, invalid_jsonl_rows = load_valid_jsonl(args.raw_jsonl)

    processed_rows = process_rows(raw_rows, tag_groups)
    pair_rows = build_relationship_pairs(json_rows, processed_rows, args.negative_ratio, args.seed)
    tag_matrix_rows, tag_matrix_fields = build_tag_matrix(processed_rows)

    output_paths = {
        "training_csv": str(args.output_dir / "manga_training.csv"),
        "training_jsonl": str(args.output_dir / "manga_training.jsonl"),
        "tag_matrix_csv": str(args.output_dir / "manga_tag_matrix.csv"),
        "weak_pairs_csv": str(args.output_dir / "weak_similarity_pairs.csv"),
        "processing_report_json": str(args.output_dir / "processing_report.json"),
    }

    write_csv(Path(output_paths["training_csv"]), processed_rows, TRAINING_FIELDS)
    write_jsonl(Path(output_paths["training_jsonl"]), processed_rows)
    write_csv(Path(output_paths["tag_matrix_csv"]), tag_matrix_rows, tag_matrix_fields)
    write_csv(Path(output_paths["weak_pairs_csv"]), pair_rows, PAIR_FIELDS)
    write_report(
        Path(output_paths["processing_report_json"]),
        raw_rows,
        processed_rows,
        json_rows,
        invalid_jsonl_rows,
        pair_rows,
        output_paths,
    )

    print(f"Processed rows: {len(processed_rows)}")
    print(f"Weak similarity pairs: {len(pair_rows)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
