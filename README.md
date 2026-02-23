# Music Recommendation App

A content-based music recommendation system built with Python and Streamlit. The app suggests songs similar to a selected track by analysing lyrical content using NLP techniques.

## Features

- Select a song from a searchable dropdown
- Get 5 similar song recommendations based on lyrical similarity
- Fast results powered by precomputed TF-IDF vectors and cosine similarity

## How It Works

1. **Text Preprocessing** — Song lyrics are cleaned and tokenized using NLTK, removing stopwords and punctuation.
2. **TF-IDF Vectorization** — Each song is converted into a numerical vector that captures the importance of words in its lyrics relative to the full dataset.
3. **Cosine Similarity** — The similarity between all songs is calculated. When a song is selected, the top 5 most similar songs are returned.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas

## Getting Started

```bash
git clone https://github.com/Oseleadeoye/my-music-app.git
cd my-music-app
pip install -r requirements.txt
streamlit run src/main.py
```

> Note: The first run will take approximately 30–60 seconds to preprocess the dataset. Subsequent runs load instantly from cache.

## Dataset

[Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) — contains song titles, artists, and full lyrics across a wide range of genres and decades.
