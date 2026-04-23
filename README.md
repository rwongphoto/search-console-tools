# Search Console Tools

Streamlit app that clusters Google Search Console queries into topics using Latent Dirichlet Allocation (LDA) and auto-labels each cluster from its most frequent terms.

## What it does

1. Upload a GSC query export (CSV).
2. Vectorize queries with `CountVectorizer` (stopwords removed via NLTK).
3. Fit an LDA topic model over the query corpus.
4. Assign each query to its dominant topic.
5. Auto-generate a topic label from the most frequent non-stopword terms.
6. Render topic-level impressions, clicks, CTR, and position aggregates with Plotly.

Useful for turning thousands of raw GSC queries into a handful of navigable topic clusters for content strategy and internal linking planning.

## Stack

- Streamlit UI
- scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`)
- NLTK (stopwords, punkt tokenizer)
- Plotly visualizations
- Heroku deployable — see `Procfile`

## Setup

```bash
pip install -r requirements.txt
streamlit run data-analysis.py
```

NLTK data (`stopwords`, `punkt`) downloads automatically on first run.

## Deploy to Heroku / Render

The included `Procfile` lets you deploy with zero extra config:

```
web: streamlit run data-analysis.py --server.port=$PORT --server.headless=true
```

## Input format

GSC Performance report CSV with `Query`, `Clicks`, `Impressions`, `CTR`, `Position`.
