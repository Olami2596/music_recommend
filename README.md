# 🎵 Lyrics-Based Music Recommender

A machine learning–driven music recommender that uses **natural language processing** (NLP) to find and suggest songs based on **lyrics similarity** and **semantic content**. Powered by the **Spotify API** for track metadata and the **Genius API** for lyrical data, this app demonstrates full-stack data science integration — from data ingestion to preprocessing, modeling, and UI delivery.

---

## 💡 Project Highlights

- 📚 **NLP-powered recommendations** using TF-IDF + cosine similarity
- 🎤 Uses real-world data from Spotify and Genius APIs
- ⚡ **Parallel lyrics processing** and caching for efficiency
- 📊 Built-in data cleaning, tokenization, and stopword removal (NLTK)
- 🎨 Responsive **Streamlit UI** with modern design and lyric explainability
- 🧠 Demonstrates skills in **ML, NLP, web APIs, and user experience design**

---

## 🧠 What This Project Demonstrates

### 🔍 Natural Language Processing
- Lyrics are preprocessed using **NLTK**, including:
  - Tokenization
  - Stopword removal
  - Regex-based cleaning
- Vectorized using **TF-IDF** (Scikit-learn)
- Recommendations ranked using **cosine similarity**

### ⚙️ Machine Learning Engineering
- Efficient pipeline for:
  - Fetching & cleaning text data
  - Vector similarity modeling
  - Performance optimization with multithreading & caching

### 🔗 Real-World Integration
- **Spotify API** for track metadata, artists, genres
- **Genius API** for lyrics extraction (with caching fallback)


---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/lyrics-recommender.git
cd lyrics-recommender
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download required NLTK data

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

### 4. Add your API credentials

Create a `.env` file:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GENIUS_ACCESS_TOKEN=your_genius_access_token
```

---

