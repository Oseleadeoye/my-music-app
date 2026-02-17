# recommend.py
import os
import joblib
import logging
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Get the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for pkl files
DF_PKL = os.path.join(BASE_DIR, 'df_cleaned.pkl')
SIM_PKL = os.path.join(BASE_DIR, 'cosine_sim.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'spotify_millsongdata.csv')


def preprocess_data():
    """Run preprocessing if pkl files don't exist."""
    logging.info("🚀 Starting preprocessing...")

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Load and sample dataset
    try:
        df = pd.read_csv(CSV_PATH).sample(10000, random_state=42)
        logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
    except Exception as e:
        logging.error("❌ Failed to load dataset: %s", str(e))
        raise e

    # Drop link column
    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

    # Text cleaning
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    logging.info("🧹 Cleaning text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    logging.info("✅ Text cleaned.")

    # Vectorization
    logging.info("🔠 Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

    # Cosine similarity
    logging.info("📐 Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logging.info("✅ Cosine similarity matrix generated.")

    # Save everything
    joblib.dump(df, DF_PKL)
    joblib.dump(cosine_sim, SIM_PKL)
    logging.info("💾 Data saved to disk.")
    logging.info("✅ Preprocessing complete.")

    return df, cosine_sim


def load_data():
    """Load data from pkl files, or preprocess first if they don't exist."""
    if os.path.exists(DF_PKL) and os.path.exists(SIM_PKL):
        logging.info("🔁 Loading cached data...")
        df = joblib.load(DF_PKL)
        cosine_sim = joblib.load(SIM_PKL)
        logging.info("✅ Data loaded successfully.")
    else:
        logging.info("⚠️ Cached data not found. Running preprocessing...")
        df, cosine_sim = preprocess_data()
    return df, cosine_sim


# Load data on import
df, cosine_sim = load_data()


def recommend_songs(song_name, top_n=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df
