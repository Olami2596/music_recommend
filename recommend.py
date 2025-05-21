import logging
import pandas as pd
from fuzzywuzzy import fuzz

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("recommend.log", encoding="utf-8"), logging.StreamHandler()]
)

def find_song_index(song_name, df, artist_name=None):
    exact_match = df[df['song'].str.lower() == song_name.lower()]
    if len(exact_match) > 0:
        if artist_name:
            artist_match = exact_match[exact_match['artist'].str.lower() == artist_name.lower()]
            if len(artist_match) > 0:
                return artist_match.index[0]
        return exact_match.index[0]

    best_ratio = 0
    best_idx = -1
    for idx, row in df.iterrows():
        ratio = fuzz.ratio(song_name.lower(), row['song'].lower())
        if artist_name:
            artist_ratio = fuzz.ratio(artist_name.lower(), row['artist'].lower())
            combined_ratio = (ratio * 0.7) + (artist_ratio * 0.3)
        else:
            combined_ratio = ratio

        if combined_ratio > best_ratio:
            best_ratio = combined_ratio
            best_idx = idx

    if best_ratio > 70:
        return best_idx
    return None

def recommend_songs(song_name, df, lyrics_sim=None, _=None, artist_name=None, top_n=5):
    logging.info(f"Recommending songs for: '{song_name}'")

    idx = find_song_index(song_name, df, artist_name)
    if idx is None or lyrics_sim is None:
        logging.warning("Song not found or no similarity matrix")
        return None

    sim_scores = list(enumerate(lyrics_sim[idx]))
    sim_scores = sorted([s for s in sim_scores if s[0] != idx], key=lambda x: x[1], reverse=True)
    top_scores = sim_scores[:top_n]
    indices = [i[0] for i in top_scores]

    result_df = df[['artist', 'song', 'id']].iloc[indices].copy()
    result_df = result_df.reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    logging.info(f"Top {len(result_df)} recommendations ready.")
    return result_df
