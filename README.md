# Music Recommendation App 🎵

An AI-powered music discovery engine that understands the soul of a song — not just its genre or popularity — and finds music that truly matches it.

## Overview

Select any song from the library, click **"Recommend Similar Songs"**, and let the model do the rest. Within seconds, you'll get 5 songs that share the same lyrical DNA as your pick.

No streaming account required. No listening history needed. Just pure, language-driven discovery.

---

## How It Works

Most recommendation systems tell you what's *popular* or what *other users* listened to. This app is different — it reads and understands the **actual words** in a song's lyrics.

### Language Vectorization
Every song's lyrics are transformed into a mathematical representation using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique highlights the words that are most unique and meaningful to each song, filtering out common filler words that carry no significance.

### Semantic Similarity
Once every song is represented as a high-dimensional vector, the model computes the **Cosine Similarity** between songs. Songs whose lyrical vectors point in the same direction — sharing similar themes, imagery, and vocabulary — are ranked as most similar. The top 5 are returned to you.

### Content-Based Filtering
Unlike collaborative filtering (which relies on other users' behaviour), this app uses **Content-Based Filtering** — it matches songs based purely on their own properties. This means it can surface deep cuts and lesser-known tracks that *feel* right, regardless of popularity.

---

## Built With

| Tool | Role |
|---|---|
| Python | Core language |
| Streamlit | Interactive web interface |
| Scikit-learn | TF-IDF vectorization & cosine similarity |
| NLTK | Text preprocessing & tokenization |
| Pandas | Dataset management |

---

## Run Locally

```bash
git clone https://github.com/Oseleadeoye/my-music-app.git
cd my-music-app
pip install -r requirements.txt
streamlit run src/main.py
```

> The first launch will take ~30–60 seconds to process the full dataset. After that, it runs instantly.

---

## Dataset

Trained on the [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) — a comprehensive collection of songs, artists, and full lyrics spanning multiple decades and genres.
