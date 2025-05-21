import streamlit as st
import pandas as pd
import re
import nltk
import logging
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from lyricsgenius import Genius
from dotenv import load_dotenv
import os
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
genius_token = os.getenv("GENIUS_ACCESS_TOKEN") or st.secrets.get("GENIUS_ACCESS_TOKEN")

# Initialize Genius client
genius = Genius(genius_token, remove_section_headers=True, skip_non_songs=True, timeout=10, retries=3)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("preprocess.log", encoding="utf-8"), logging.StreamHandler()]
)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text)).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# --- Caching Lyrics ---
def fetch_lyrics(song_name, artist_name, retries=3):
    logging.info(f"Fetching lyrics for {song_name} by {artist_name}")
    clean_song_name = re.sub(r'\([^)]*\)', '', song_name).strip()

    for attempt in range(retries):
        try:
            song = genius.search_song(clean_song_name, artist_name)
            if not song and ' - ' in clean_song_name:
                main_title = clean_song_name.split(' - ')[0].strip()
                song = genius.search_song(main_title, artist_name)
            if not song:
                song = genius.search_song(clean_song_name)
            lyrics = ""
            if song:
                lyrics = re.sub(r'\[.*?\]', '', song.lyrics)
                lyrics = re.sub(r'\d+Embed$', '', lyrics)
                lyrics = re.sub(r'Embed$', '', lyrics)
                return lyrics
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)
    logging.warning(f"Could not fetch lyrics for {song_name} by {artist_name}")
    return ""

def fetch_lyrics_cached(song_name, artist_name):
    cache_dir = Path("lyrics_cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5(f"{song_name}_{artist_name}".encode()).hexdigest()
    cache_file = cache_dir / f"{key}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    lyrics = fetch_lyrics(song_name, artist_name)
    cache_file.write_text(lyrics, encoding="utf-8")
    return lyrics

def get_artist_genres(artist_id, sp):
    try:
        artist = sp.artist(artist_id)
        return artist.get('genres', [])
    except Exception as e:
        logging.warning(f"Failed to get genres for artist {artist_id}: {str(e)}")
        return []

def preprocess_and_vectorize(track_info, sp):
    logging.info(f"Starting preprocessing for {track_info['name']} by {track_info['artists'][0]['name']}")

    source_lyrics = fetch_lyrics_cached(track_info["name"], track_info["artists"][0]["name"])

    data = {
        "song": [track_info["name"]],
        "artist": [track_info["artists"][0]["name"]],
        "lyrics": [source_lyrics],
        "id": [track_info["id"]],
        "popularity": [track_info.get("popularity", 0)]
    }

    # Related Tracks Collection
    related_tracks = []

    # Same artist
    try:
        search_query = f"artist:{track_info['artists'][0]['name']}"
        artist_tracks = sp.search(q=search_query, type="track", limit=20, market="US")["tracks"]["items"]
        related_tracks.extend(artist_tracks)
        logging.info(f"✅ {len(artist_tracks)} artist tracks fetched")
    except Exception as e:
        logging.warning(f"⚠️ Artist track fetch failed: {str(e)}")

    # Genre-based tracks
    try:
        artist_genres = get_artist_genres(track_info['artists'][0]['id'], sp)
        if artist_genres:
            genre_query = f"genre:{artist_genres[0]}"
            genre_tracks = sp.search(q=genre_query, type="track", limit=50, market="US")["tracks"]["items"]
            sampled_genre_tracks = random.sample(genre_tracks, min(10, len(genre_tracks)))
            related_tracks.extend(sampled_genre_tracks)
            logging.info(f"✅ {len(sampled_genre_tracks)} genre tracks fetched from genre '{artist_genres[0]}'")
    except Exception as e:
        logging.warning(f"⚠️ Genre-based search failed: {str(e)}")

    # Deduplicate
    track_ids = set()
    unique_tracks = []
    for track in related_tracks:
        if track['id'] != track_info['id'] and track['id'] not in track_ids:
            track_ids.add(track['id'])
            unique_tracks.append(track)

    unique_tracks = unique_tracks[:30]
    logging.info(f"📦 {len(unique_tracks)} unique tracks selected for processing")

    # Parallel lyrics fetching
    def process_track(track):
        song_name = track["name"]
        artist_name = track["artists"][0]["name"]
        lyrics = fetch_lyrics_cached(song_name, artist_name)
        return {
            "song": song_name,
            "artist": artist_name,
            "lyrics": lyrics,
            "id": track["id"],
            "popularity": track.get("popularity", 0)
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_track, unique_tracks))

    for r in results:
        data["song"].append(r["song"])
        data["artist"].append(r["artist"])
        data["lyrics"].append(r["lyrics"])
        data["id"].append(r["id"])
        data["popularity"].append(r["popularity"])

    df = pd.DataFrame(data)
    logging.info(f"✅ Dataset created with {len(df)} tracks")

    # Clean lyrics
    df['cleaned_lyrics'] = df['lyrics'].apply(preprocess_text)
    valid_lyrics_count = df['cleaned_lyrics'].str.strip().str.len().gt(0).sum()
    if valid_lyrics_count < 2:
        logging.warning("Not enough valid lyrics for similarity calculation")
        return df, None, None, source_lyrics

    # Compute lyrics similarity
    try:
        tfidf = TfidfVectorizer(max_features=5000, min_df=2)
        tfidf_matrix = tfidf.fit_transform(df['cleaned_lyrics'])
        lyrics_sim = cosine_similarity(tfidf_matrix)
        logging.info("✅ Lyrics similarity matrix generated")
    except Exception as e:
        logging.error(f"Error computing lyrics similarity: {str(e)}")
        lyrics_sim = None

    return df, lyrics_sim, None, source_lyrics
