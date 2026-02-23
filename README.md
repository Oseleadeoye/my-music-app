# Music Recommendation App 🎵

A machine learning-based content recommendation engine that suggests similar songs based on their lyrics. Built with Python, Scikit-learn, and Streamlit.

## 🚀 Features

- **Content-Based Filtering**: Uses TF-IDF vectorization and Cosine Similarity to find songs with similar lyrical content.
- **Instant Recommendations**: Select a song and get top 5 similar tracks immediately.
- **Auto-Preprocessing**: The app automatically processes the raw dataset on the first run, making it easy to deploy without uploading large pre-calculated files.
- **Interactive UI**: Clean and simple web interface powered by Streamlit.

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** (Web App Framework)
- **Pandas** (Data Manipulation)
- **Scikit-learn** (Machine Learning / Vectorization)
- **NLTK** (Natural Language Processing)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Oseleadeoye/my-music-app.git
   cd my-music-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run src/main.py
   ```
   *Note: The first run might take ~30-60 seconds to preprocess the data. Subsequent runs will be instant.*

## ☁️ Deployment

This app is ready for **Streamlit Community Cloud**:

1. Push your code to GitHub.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app, select this repository, and set the main file path to `src/main.py`.
4. Click **Deploy**!

## 📂 Project Structure

```
├── src/
│   ├── main.py          # Streamlit application UI
│   ├── recommend.py     # Recommendation logic & preprocessing
│   └── spotify_...csv   # Dataset
├── .gitignore           # Ignored files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 📝 Credits

Original dataset: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
