
import pickle
from pathlib import Path

import requests
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "movie_list.pkl", "rb") as f:
    movies = pickle.load(f)

with open(BASE_DIR / "similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b0b0f;
    color: #f5f5f7;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(11, 11, 15, 0.95);
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 42px 38px;
    border: 1px solid #30202a;
    border-radius: 24px;
    background: radial-gradient(ellipse at top right, #54202d 0%, #21131d 40%, #111116 78%);
    margin-bottom: 30px;
}

.eyebrow {
    color: #ff647c;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(36px, 5vw, 60px);
    line-height: 1.08;
    margin: 14px 0;
    color: #ffffff;
    font-weight: 800;
}

.hero p {
    color: #c4bdc5;
    font-size: 16px;
    max-width: 620px;
    line-height: 1.7;
}

.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 700;
    margin: 20px 0 6px;
    color: #ffffff;
}

.muted {
    color: #a5a1ac;
    font-size: 14px;
}

.poster-card {
    background: #15151c;
    border: 1px solid #292932;
    border-radius: 16px;
    padding: 10px;
    height: 100%;
    transition: border-color 0.2s ease;
}

.poster-card:hover {
    border-color: #ff647c;
}

.movie-title {
    font-family: 'Outfit', sans-serif;
    color: #f7f5f8;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.4;
    padding: 12px 3px 5px;
}

.rank {
    color: #ff647c;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 5px 3px 0;
}

div.stButton > button {
    background: linear-gradient(110deg, #ff4968, #d92750);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border: 1px solid #ff8ca0;
    color: white;
    filter: brightness(1.1);
}

div[data-baseweb="select"] > div {
    background: #17171f;
    border-color: #383842;
    border-radius: 12px;
    color: white;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    border-top: 1px solid #292932;
    margin-top: 60px;
    padding-top: 22px;
    text-align: center;
    color: #777580;
    font-size: 12px;
}

@media (max-width: 768px) {
    .hero {
        padding: 28px 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_models():
    if not movie_path.exists():
        raise FileNotFoundError(
            f"Movie data file not found. Checked: {BASE_DIR / 'movie_list.pkl'} "
            f"and {MODEL_DIR / 'movie_list.pkl'}"
        )

    if not similarity_path.exists():
        raise FileNotFoundError(
            f"Similarity file not found. Checked: {BASE_DIR / 'similarity.pkl'} "
            f"and {MODEL_DIR / 'similarity.pkl'}"
        )

    with movie_path.open("rb") as file:
        movie_data = pickle.load(file)

    with similarity_path.open("rb") as file:
        similarity_data = pickle.load(file)

    return movie_data, similarity_data


try:
    movies, similarity = load_models()
except Exception as error:
    st.error("Unable to load the movie recommendation data.")
    st.code(str(error))
    st.stop()


# ---------------- TMDB POSTERS ----------------
# Add TMDB_API_KEY to Streamlit Cloud Secrets.
# Missing API keys or unavailable posters won't stop recommendations.
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    TMDB_API_KEY = ""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US"
            },
            timeout=8
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (requests.RequestException, ValueError):
        pass

    return None


# ---------------- RECOMMENDATIONS ----------------
def recommend(movie_title, number=5):
    matches = movies.index[movies["title"] == movie_title].tolist()

    if not matches:
        return []

    index = matches[0]

    distances = sorted(
        enumerate(similarity[index]),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for movie_index, score in distances:
        if movie_index == index:
            continue

        row = movies.iloc[movie_index]

        results.append({
            "title": row["title"],
            "movie_id": row["movie_id"],
            "score": float(score)
        })

        if len(results) == number:
            break

    return results


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL MOVIE DISCOVERY ENGINE</div>
    <h1>Find your next<br>favourite film<span style="color:#ff647c">.</span></h1>
    <p>
        Not sure what to watch tonight? Choose a movie you love.
        Our recommendation engine will find five similar films
        worth adding to your watchlist.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- SEARCH SECTION ----------------
st.markdown(
    '<div class="section-title">🎬 Discover movies</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="muted">Search the collection and choose a movie to get started.</div>',
    unsafe_allow_html=True
)

movie_list = movies["title"].dropna().astype(str).tolist()

if not movie_list:
    st.warning("No movie titles were found in the movie dataset.")
    st.stop()

selected_movie = st.selectbox(
    "Choose your movie",
    movie_list,
    index=0
)

left, right = st.columns([1, 3])

with left:
    show_recommendations = st.button(
        "✦  Find Similar Movies",
        use_container_width=True
    )

with right:
    st.markdown(
        '<div class="muted" style="padding-top:14px;">'
        'Personalized picks based on movie similarity</div>',
        unsafe_allow_html=True
    )


# ---------------- RESULTS ----------------
if show_recommendations:
    recommendations = recommend(selected_movie)

    if not recommendations:
        st.warning("No recommendations were found for this movie.")
        st.stop()

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">Because you selected '
        f'<span style="color:#ff647c">{selected_movie}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Your next five movies to explore.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Finding movies and loading posters..."):
        for start in range(0, len(recommendations), 5):
            batch = recommendations[start:start + 5]
            columns = st.columns(5, gap="medium")

            for position, (column, movie) in enumerate(
                zip(columns, batch), start=start + 1
            ):
                with column:
                    poster = fetch_poster(movie["movie_id"])

                    st.markdown(
                        f'<div class="poster-card">'
                        f'<div class="rank">PICK {position:02d}</div>',
                        unsafe_allow_html=True
                    )

                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:240px;display:flex;'
                            'align-items:center;justify-content:center;'
                            'background:#24242d;border-radius:10px;'
                            'color:#aaa;font-size:40px;">🎞️</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{movie["title"]}</div></div>',
                        unsafe_allow_html=True
                    )

    st.success("Your recommendations are ready!")

    if not TMDB_API_KEY:
        st.caption(
            "Posters are optional. Add your TMDB API key to Streamlit "
            "Secrets to display movie posters."
        )


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    CINEMATCH · MOVIE RECOMMENDATION SYSTEM<br><br>
    Discover something worth watching.
</div>
""", unsafe_allow_html=True)
