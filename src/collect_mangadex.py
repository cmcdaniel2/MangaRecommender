import os
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

API_BASE = "https://api.mangadex.org"
TOKEN_URL = "https://auth.mangadex.org/realms/mangadex/protocol/openid-connect/token"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_JSONL_PATH = RAW_DIR / "mangadex_manga_raw.jsonl"
RAW_CSV_PATH = RAW_DIR / "mangadex_manga_raw.csv"
TAGS_JSON_PATH = RAW_DIR / "mangadex_tags.json"


def get_headers(access_token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": os.getenv("MANGADEX_USER_AGENT", "MangaRecNLP/0.1"),
        "Accept": "application/json",
    }

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    return headers


def get_access_token() -> str:
    """
    Only needed for authenticated MangaDex routes.

    For your raw recommender dataset, you can usually skip this and use
    the public /manga endpoint anonymously.
    """

    client_id = os.getenv("MANGADEX_CLIENT_ID")
    client_secret = os.getenv("MANGADEX_CLIENT_SECRET")
    username = os.getenv("MANGADEX_USERNAME")
    password = os.getenv("MANGADEX_PASSWORD")

    missing = [
        name for name, value in {
            "MANGADEX_CLIENT_ID": client_id,
            "MANGADEX_CLIENT_SECRET": client_secret,
            "MANGADEX_USERNAME": username,
            "MANGADEX_PASSWORD": password,
        }.items()
        if not value
    ]

    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    payload = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    response = requests.post(TOKEN_URL, data=payload, timeout=30)
    response.raise_for_status()

    token_data = response.json()
    return token_data["access_token"]


def request_json(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"

    for attempt in range(max_retries):
        response = requests.get(
            url,
            params=params,
            headers=get_headers(access_token),
            timeout=30,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else 2 ** attempt
            print(f"Rate limited. Sleeping {wait} seconds...")
            time.sleep(wait)
            continue

        if 500 <= response.status_code < 600:
            wait = 2 ** attempt
            print(f"Server error {response.status_code}. Sleeping {wait} seconds...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Failed request after {max_retries} retries: {url}")


def pick_localized_text(obj: Dict[str, str], preferred_lang: str = "en") -> str:
    """
    MangaDex stores title/description as localized dictionaries.
    Prefer English, otherwise fall back to the first available value.
    """

    if not isinstance(obj, dict) or not obj:
        return ""

    if preferred_lang in obj and obj[preferred_lang]:
        return obj[preferred_lang]

    for value in obj.values():
        if value:
            return value

    return ""


def flatten_alt_titles(alt_titles: List[Dict[str, str]]) -> str:
    titles = []

    if not isinstance(alt_titles, list):
        return ""

    for title_obj in alt_titles:
        if isinstance(title_obj, dict):
            titles.extend([value for value in title_obj.values() if value])

    return " | ".join(sorted(set(titles)))


def extract_relationship_names(relationships: List[Dict[str, Any]], rel_type: str) -> str:
    """
    If you request includes[]=author / includes[]=artist / includes[]=cover_art,
    MangaDex often includes relationship attributes.
    """

    names = []

    for rel in relationships:
        if rel.get("type") != rel_type:
            continue

        attrs = rel.get("attributes", {})

        if rel_type in ["author", "artist"]:
            name = attrs.get("name")
            if name:
                names.append(name)

        elif rel_type == "cover_art":
            filename = attrs.get("fileName")
            if filename:
                names.append(filename)

    return " | ".join(sorted(set(names)))


def flatten_manga_item(item: Dict[str, Any]) -> Dict[str, Any]:
    attrs = item.get("attributes", {})
    relationships = item.get("relationships", [])

    tags = attrs.get("tags", [])
    tag_names = []
    tag_groups = []

    for tag in tags:
        tag_attrs = tag.get("attributes", {})
        tag_name = pick_localized_text(tag_attrs.get("name", {}), preferred_lang="en")
        tag_group = tag_attrs.get("group", "")

        if tag_name:
            tag_names.append(tag_name)
        if tag_group:
            tag_groups.append(tag_group)

    title = pick_localized_text(attrs.get("title", {}), preferred_lang="en")
    description = pick_localized_text(attrs.get("description", {}), preferred_lang="en")

    return {
        "mangadex_id": item.get("id", ""),
        "title": title,
        "alt_titles": flatten_alt_titles(attrs.get("altTitles", [])),
        "description": description,
        "original_language": attrs.get("originalLanguage", ""),
        "last_volume": attrs.get("lastVolume", ""),
        "last_chapter": attrs.get("lastChapter", ""),
        "publication_demographic": attrs.get("publicationDemographic", ""),
        "status": attrs.get("status", ""),
        "year": attrs.get("year", ""),
        "content_rating": attrs.get("contentRating", ""),
        "state": attrs.get("state", ""),
        "created_at": attrs.get("createdAt", ""),
        "updated_at": attrs.get("updatedAt", ""),
        "tags": ", ".join(sorted(set(tag_names))),
        "tag_groups": ", ".join(sorted(set(tag_groups))),
        "authors": extract_relationship_names(relationships, "author"),
        "artists": extract_relationship_names(relationships, "artist"),
        "cover_file": extract_relationship_names(relationships, "cover_art"),
    }


def fetch_tags(access_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Useful reference file. The /manga response usually already includes tag info,
    but saving the tag list helps with debugging and filtering.
    """

    payload = request_json("manga/tag", access_token=access_token)

    with open(TAGS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved tags to {TAGS_JSON_PATH}")
    return payload


def fetch_manga_dataset(
    max_records: int = 5000,
    limit: int = 100,
    sleep_seconds: float = 0.4,
    access_token: Optional[str] = None,
    content_ratings: Optional[List[str]] = None,
    translated_languages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch raw manga metadata from MangaDex.

    This saves:
    1. Raw JSONL: data/raw/mangadex_manga_raw.jsonl
    2. Flattened CSV: data/raw/mangadex_manga_raw.csv
    """

    content_ratings = content_ratings or ["safe", "suggestive"]
    translated_languages = translated_languages or ["en"]

    all_rows = []
    total = None
    offset = 0

    if RAW_JSONL_PATH.exists():
        RAW_JSONL_PATH.unlink()

    pbar = tqdm(total=max_records, desc="Fetching MangaDex manga")

    while offset < max_records:
        current_limit = min(limit, max_records - offset)

        params = {
            "limit": current_limit,
            "offset": offset,

            # Include relationship attributes in the response
            "includes[]": ["author", "artist", "cover_art"],

            # Good default for a recommender dataset
            "contentRating[]": content_ratings,
            "availableTranslatedLanguage[]": translated_languages,

            # Makes the dataset more useful by prioritizing known titles
            "order[followedCount]": "desc",

            # Avoid entries with no readable chapters if possible
            "hasAvailableChapters": "true",
        }

        payload = request_json("manga", params=params, access_token=access_token)

        data = payload.get("data", [])
        if total is None:
            total = payload.get("total", max_records)

        if not data:
            break

        with open(RAW_JSONL_PATH, "a", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        flattened = [flatten_manga_item(item) for item in data]
        all_rows.extend(flattened)

        fetched = len(data)
        offset += fetched
        pbar.update(fetched)

        if offset >= total:
            break

        time.sleep(sleep_seconds)

    pbar.close()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["mangadex_id"])

    df.to_csv(RAW_CSV_PATH, index=False)

    print(f"\nSaved raw JSONL to: {RAW_JSONL_PATH}")
    print(f"Saved flattened CSV to: {RAW_CSV_PATH}")
    print(f"Rows saved: {len(df)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Collect raw MangaDex manga metadata")

    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.4)

    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use MangaDex OAuth credentials from .env",
    )

    parser.add_argument(
        "--content-ratings",
        nargs="+",
        default=["safe", "suggestive"],
        help="Example: safe suggestive",
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Example: en ja",
    )

    parser.add_argument(
        "--fetch-tags",
        action="store_true",
        help="Also save the MangaDex tag list",
    )

    args = parser.parse_args()

    access_token = None

    if args.auth:
        print("Authenticating with MangaDex...")
        access_token = get_access_token()
        print("Authenticated.")

    if args.fetch_tags:
        fetch_tags(access_token=access_token)

    fetch_manga_dataset(
        max_records=args.max_records,
        limit=args.limit,
        sleep_seconds=args.sleep,
        access_token=access_token,
        content_ratings=args.content_ratings,
        translated_languages=args.languages,
    )


if __name__ == "__main__":
    main()