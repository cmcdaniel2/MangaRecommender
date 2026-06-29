# MangaRec

MangaRec is a content-based manga recommendation prototype built from MangaDex metadata. It uses NLP over titles, descriptions, tags, creators, demographics, and related metadata to recommend similar manga.

The project currently includes:

- A MangaDex collection script for raw manga metadata.
- A preprocessing pipeline that turns raw exports into model-ready training data.
- A TF-IDF cosine-similarity recommender baseline.
- A sentence-transformer embedding recommender for semantic similarity.
- Sample recommendation outputs for quick sanity checks.

## Project Status

This is an early recommendation-system prototype. It is designed to answer:

> Given a manga, what other manga are textually or semantically similar?

The current models are content-based only. They do not yet use user ratings, clicks, favorites, reading history, or collaborative filtering.

## Repository Structure

```text
MangaRec/
  data/
    raw/
      mangadex_manga_raw.csv
      mangadex_manga_raw.jsonl
      mangadex_tags.json
    processed/
      manga_training.csv
      manga_training.jsonl
      manga_tag_matrix.csv
      weak_similarity_pairs.csv
      processing_report.json
    models/
      tfidf_baseline/
      sentence_transformer_baseline/
  src/
    collect_mangadex.py
    process_mangadex.py
    baseline_tfidf_recommender.py
    sentence_transformer_recommender.py
  requirements.txt
  README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The project has been run with Python 3.11 on Windows.

## Data Collection

Raw MangaDex data can be collected with:

```powershell
.venv\Scripts\python.exe src\collect_mangadex.py --max-records 2500 --fetch-tags
```

By default, the collector writes:

- `data/raw/mangadex_manga_raw.jsonl`
- `data/raw/mangadex_manga_raw.csv`
- `data/raw/mangadex_tags.json`

The collector uses public MangaDex endpoints by default. Optional authenticated routes can use credentials from `.env`.

Example `.env` keys:

```text
MANGADEX_USER_AGENT=MangaRecNLP/0.1
MANGADEX_CLIENT_ID=
MANGADEX_CLIENT_SECRET=
MANGADEX_USERNAME=
MANGADEX_PASSWORD=
```

## Data Processing

Process the raw export into model-ready training data:

```powershell
.venv\Scripts\python.exe src\process_mangadex.py
```

This creates:

- `data/processed/manga_training.csv`
- `data/processed/manga_training.jsonl`
- `data/processed/manga_tag_matrix.csv`
- `data/processed/weak_similarity_pairs.csv`
- `data/processed/processing_report.json`

The main modeling field is `model_text`, which combines the title, alternate titles, cleaned description, genres, themes, formats, demographic, authors, and artists.

Current processed corpus summary:

- 2,500 manga rows
- 75 observed MangaDex tags
- 2,500 non-empty `model_text` rows
- 112 weak positive relationship pairs
- 112 sampled weak negative pairs

## Model 1: TF-IDF Cosine Similarity

The TF-IDF model is a fast lexical baseline. It works well when similar manga share important words such as `hunter`, `level`, `isekai`, `school`, `revenge`, or specific genre terms.

Build the model:

```powershell
.venv\Scripts\python.exe src\baseline_tfidf_recommender.py build
```

Recommend similar manga:

```powershell
.venv\Scripts\python.exe src\baseline_tfidf_recommender.py recommend "Solo Leveling" --top-k 10
```

Save recommendations to CSV:

```powershell
.venv\Scripts\python.exe src\baseline_tfidf_recommender.py recommend "Solo Leveling" --top-k 10 --output data\models\tfidf_baseline\sample_solo_leveling_recommendations.csv
```

Generated artifacts:

- `data/models/tfidf_baseline/tfidf_vectorizer.joblib`
- `data/models/tfidf_baseline/tfidf_matrix.npz`
- `data/models/tfidf_baseline/manga_metadata.csv`
- `data/models/tfidf_baseline/baseline_config.json`

Current baseline size:

- 2,500 manga vectors
- 24,889 TF-IDF features

## Model 2: Sentence-Transformer Embeddings

The sentence-transformer model is a semantic baseline. It embeds each manga's `model_text` into a dense vector, then uses cosine similarity to find neighbors. This can catch similarity even when two manga use different wording.

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Build embeddings:

```powershell
.venv\Scripts\python.exe src\sentence_transformer_recommender.py build
```

Recommend similar manga by title or MangaDex ID:

```powershell
.venv\Scripts\python.exe src\sentence_transformer_recommender.py recommend "Solo Leveling" --top-k 10
```

Search by free-text description:

```powershell
.venv\Scripts\python.exe src\sentence_transformer_recommender.py search "dark fantasy revenge story with monsters" --top-k 10
```

Generated artifacts:

- `data/models/sentence_transformer_baseline/manga_embeddings.npy`
- `data/models/sentence_transformer_baseline/manga_metadata.csv`
- `data/models/sentence_transformer_baseline/embedding_config.json`

Current embedding baseline size:

- 2,500 manga embeddings
- 384 dimensions per embedding
- L2-normalized vectors for cosine similarity

## Example Results

For the query `Solo Leveling`, the sentence-transformer baseline returns neighbors such as:

| Rank | Title | Cosine Similarity |
| ---: | --- | ---: |
| 1 | Leveling Up Alone | 0.6051 |
| 2 | Level Up with the Gods | 0.5624 |
| 3 | LV999 no Murabito | 0.5587 |

For the free-text query `dark fantasy revenge story with monsters`, sample matches include:

| Rank | Title | Cosine Similarity |
| ---: | --- | ---: |
| 1 | Ikenie ni Natta Ore ga Nazeka Jashin wo Horoboshite Shimatta Ken | 0.5492 |
| 2 | Cheat Skill "Shisha Sosei" ga Kakusei Shite, Inishie no Maougun wo Fukkatsu Sasete Shimaimashita ~Dare mo Shinasenai Saikyou Hiiro~ | 0.5364 |
| 3 | Garbage Brave: Isekai ni Shoukan Sare Suterareta Yuusha no Fukushuu Monogatari | 0.5332 |

## How Recommendations Work

1. Raw MangaDex metadata is collected.
2. The processor cleans descriptions, removes links/markup, groups tags, and builds `model_text`.
3. A model converts each manga's `model_text` into a vector.
4. Query manga are looked up by MangaDex ID, title, partial title, or alternate title.
5. Cosine similarity ranks the nearest manga vectors.

For normalized vectors, cosine similarity is computed as a dot product.

## Evaluation Data

`data/processed/weak_similarity_pairs.csv` contains lightweight pair labels:

- Positive pairs from in-corpus MangaDex relationships such as sequel, prequel, shared universe, spin-off, and alternate version.
- Negative pairs sampled from manga with no or low tag overlap.

These labels are not perfect ground truth, but they are useful for:

- Comparing TF-IDF vs sentence-transformer retrieval.
- Measuring whether known related manga are ranked highly.
- Creating a first offline evaluation loop.

## Next Steps

Recommended next improvements:

1. Add an evaluation script using `weak_similarity_pairs.csv`.
2. Compare TF-IDF and sentence-transformer results with ranking metrics like recall@10 and mean reciprocal rank.
3. Build a hybrid recommender that combines sentence embeddings, tag overlap, demographic filters, and status/content-rating filters.
4. Add a small Streamlit UI for search and recommendations.
5. Add user behavior data such as favorites, ratings, clicks, skips, and reading history.
6. Train or fine-tune a model using real user preference data once available.

## Notes

- The first sentence-transformer build may download model weights from Hugging Face.
- On Windows, Hugging Face may warn about symlink caching. The model still works; it may just use more disk space.
- The raw MangaDex data should be regenerated locally and kept in line with MangaDex API usage rules.
- `.env` should not be committed.

## License

Add a license before publishing this repository publicly.
