import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from preprocess import preprocess_and_vectorize, fetch_lyrics, get_artist_genres
from recommend import recommend_songs
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID") or st.secrets.get("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or st.secrets.get("SPOTIFY_CLIENT_SECRET")


# Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10)

# Page config
st.set_page_config(
    page_title="Lyrics Music Recommender 🎵", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log", encoding="utf-8")]
)

# 🧼 Modern CSS
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #168f3e;
        transition: background-color 0.2s ease;
    }
    .recommend-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .recommend-card h4 {
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        font-size: 1.25rem;
    }
    .recommend-card a {
        color: #1DB954;
        font-weight: bold;
        text-decoration: none;
    }
    .recommend-card a:hover {
        text-decoration: underline;
    }
    @media (max-width: 768px) {
        .stColumns {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎵 Lyrics Recommender")
    st.markdown("This app recommends songs based on lyrics similarity and genre, using Spotify and Genius APIs.")
    recommendation_count = st.slider("Number of recommendations", 3, 10, 5)

# Session state
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "tracks" not in st.session_state:
    st.session_state.tracks = []

# Search options
search_type = st.radio("Search by:", ["Song Title", "Artist", "Lyrics Fragment"], horizontal=True)
search_placeholder = {
    "Song Title": "Enter song title (e.g., Shape of You)",
    "Artist": "Enter artist name (e.g., Adele)",
    "Lyrics Fragment": "Enter lyric (e.g., I see fire)"
}
search_query = st.text_input("🔍 Search", placeholder=search_placeholder[search_type])

if st.button("🚀 Find Songs"):
    if not search_query:
        st.warning("Please enter a search term.")
    else:
        with st.spinner("Searching..."):
            try:
                query = f"artist:{search_query}" if search_type == "Artist" else f"track:{search_query}" if search_type == "Song Title" else search_query
                results = sp.search(q=query, type="track", limit=10, market="US")
                st.session_state.tracks = results["tracks"]["items"]
                st.session_state.search_results = [
                    f"{t['name']} by {t['artists'][0]['name']}" for t in st.session_state.tracks
                ]
            except Exception as e:
                st.error(f"Search failed: {e}")

# Song selection and display
if st.session_state.search_results:
    selected_track_name = st.selectbox("Select a song for recommendations:", st.session_state.search_results)
    selected_index = st.session_state.search_results.index(selected_track_name)
    selected_track_info = st.session_state.tracks[selected_index]

    col1, col2 = st.columns([1, 3])
    with col1:
        if selected_track_info["album"]["images"]:
            st.image(selected_track_info["album"]["images"][0]["url"], width=150)
    with col2:
        st.markdown(f"### {selected_track_info['name']}")
        st.markdown(f"**Artist:** {selected_track_info['artists'][0]['name']}")
        st.markdown(f"**Album:** {selected_track_info['album']['name']}")
        genres = get_artist_genres(selected_track_info['artists'][0]['id'], sp)
        if genres:
            st.markdown(f"**Genre:** {genres[0].title()}")
        if selected_track_info['external_urls'].get('spotify'):
            st.markdown(f"[🔗 Open in Spotify]({selected_track_info['external_urls']['spotify']})")

    source_lyrics = fetch_lyrics(
        selected_track_info["name"],
        selected_track_info["artists"][0]["name"]
    )

    if source_lyrics:
        with st.expander("📜 Show Lyrics"):
            st.markdown(
                f'<div style="background:#f9f9f9;padding:15px;border-radius:10px;">{source_lyrics.replace("\n", "<br>")}</div>',
                unsafe_allow_html=True
            )

    if st.button("🔍 Generate Recommendations"):
        with st.spinner("Generating..."):
            df, lyrics_sim, _, _ = preprocess_and_vectorize(selected_track_info, sp)

            if df is None:
                st.error("Failed to analyze track.")
            else:
                recommendations = recommend_songs(
                    selected_track_info["name"],
                    df,
                    lyrics_sim,
                    None,
                    selected_track_info["artists"][0]["name"],
                    top_n=recommendation_count
                )

                if not recommendations.empty:
                    st.markdown("### 🎯 Recommended Songs")
                    for _, row in recommendations.iterrows():
                        track = sp.track(row['id'])
                        image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
                        artist_id = track['artists'][0]['id']
                        genres = get_artist_genres(artist_id, sp)
                        genre_name = genres[0].title() if genres else "Unknown"
                        link = f"https://open.spotify.com/track/{row['id']}"
                        lyrics = fetch_lyrics(row['song'], row['artist'])

                        st.markdown(f"""
                            <div class="recommend-card">
                                <div style="display:flex;flex-wrap:wrap;">
                                    <div style="flex:1;min-width:120px;">
                                        {'<img src="'+image_url+'" width="120"/>' if image_url else ''}
                                    </div>
                                    <div style="flex:3;padding-left:20px;min-width:200px;">
                                        <h4>{row['song']}</h4>
                                        <p><strong>Artist:</strong> {row['artist']}</p>
                                        <p><strong>Genre:</strong> {genre_name}</p>
                                        <a href="{link}" target="_blank">Open in Spotify</a>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        if lyrics:
                            with st.expander("📜 Show Lyrics"):
                                st.markdown(
                                    f'<div style="background:#f9f9f9;padding:15px;border-radius:10px;">{lyrics.replace("\n", "<br>")}</div>',
                                    unsafe_allow_html=True
                                )

                else:
                    st.warning("No recommendations found.")

# Footer
st.markdown("""
<hr>
<div style='text-align:center;font-size:0.8rem;color:#888'>
    Music Recommender App © 2025 — Powered by Spotify and Genius APIs
</div>
""", unsafe_allow_html=True)
