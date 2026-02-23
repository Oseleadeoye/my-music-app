# 🎵 Music Recommendation App

A content-based music recommendation system that suggests songs similar to the one you love — powered by machine learning and song lyrics.

## 🎧 What It Does

Simply pick a song from the dropdown, hit **"Recommend Similar Songs"**, and get a list of 5 songs with similar lyrical themes and style — instantly.

No sign-up. No account. Just music.

## 🧠 How It Works

The app uses **Natural Language Processing (NLP)** and **Machine Learning** to analyse the lyrics of over 57,000 songs. When you select a song:

1. The lyrics are vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency)
2. **Cosine Similarity** is calculated between songs to measure how alike their lyrics are
3. The top 5 most similar songs are returned to you

This is called **Content-Based Filtering** — it recommends based on the actual content of the song, not other users' listening habits.

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app interface |
| Scikit-learn | TF-IDF & Cosine Similarity |
| NLTK | Text cleaning & tokenization |
| Pandas | Data handling |

## 📂 Run Locally

```bash
git clone https://github.com/Oseleadeoye/my-music-app.git
cd my-music-app
pip install -r requirements.txt
streamlit run src/main.py
```

> ⚠️ The first run will take ~30–60 seconds to process the dataset. After that it's instant.

## 📊 Dataset

The app uses the [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) — a large collection of songs with their lyrics, artists, and metadata.
