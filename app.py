# app_v4.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from difflib import get_close_matches
import random

# ===================================================================
# FUNÇÕES DE BACKEND (CARREGAMENTO, PROCESSAMENTO, RECOMENDAÇÃO)
# ===================================================================

@st.cache_data
def load_and_process_data(filepath="top_10000_1960-now.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None, None, None
    
    df = df.copy()
    rename_map = {
        'Track Name': 'track', 'Track URI': 'track_uri', 'Artist Name(s)': 'artist',
        'Album Name': 'album', 'Album Release Date': 'year', 'Album Image URL': 'image_url', 
        'Artist Genres': 'artist_genres', 'Track Duration (ms)': 'duration_ms', 'Explicit': 'explicit', 
        'Key': 'key', 'Mode': 'mode', 'Time Signature': 'time_signature'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if 'year' in df.columns:
        df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
        df.dropna(subset=['year'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['year'] = df['year'].astype(int)

    text_cols = ['track','artist','album','artist_genres']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    num_features = ['Danceability','Energy','Loudness','Speechiness','Acousticness',
                    'Instrumentalness','Liveness','Valence','Tempo', 'duration_ms']
    cat_features = ['key', 'mode', 'time_signature']
    
    df[num_features] = df[num_features].fillna(0)
    df[cat_features] = df[cat_features].fillna(0)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough', sparse_threshold=0
    )
    
    feature_matrix = preprocessor.fit_transform(df[num_features + cat_features])
    
    if 'artist_genres' in df.columns:
        top_genres = df['artist_genres'].str.split(', ').explode().str.strip().value_counts().head(40).index.tolist()
        genres = sorted(top_genres)
    else:
        genres = []

    return df, feature_matrix, genres

@st.cache_data
def build_tfidf_matrix(_df):
    text_data = (_df['track'].astype(str) + ' ' + _df['artist'].astype(str) + ' ' +
                 _df['album'].astype(str) + ' ' + _df['artist_genres'].astype(str))
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.75)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, tfidf_matrix

def get_recommendations(df, feature_matrix, vectorizer, tfidf_matrix, query, top_n,
                        weight_text, fuzzy_threshold, mode):
    
    seed_profile = {'text': None, 'features': None, 'info': None, 'indices_to_ignore': []}
    search_col = mode
    candidates = df[search_col].unique().tolist()
    matches = get_close_matches(query, candidates, n=1, cutoff=fuzzy_threshold)
    if not matches: return [], None
    matched_item = matches[0]
    
    if mode == 'track':
        idx = df.index[df['track'] == matched_item].tolist()[0]
        seed_profile['info'] = df.loc[idx]
        seed_profile['text'] = tfidf_matrix[idx]
        seed_profile['features'] = feature_matrix[idx]
        seed_profile['indices_to_ignore'].append(idx)
    else:
        subset_df = df[df[search_col] == matched_item]
        item_indices = subset_df.index
        
        seed_profile['info'] = subset_df.iloc[0]
        seed_profile['features'] = feature_matrix[item_indices].mean(axis=0)
        text_data = ' '.join(subset_df['track'] + ' ' + subset_df['artist'] + ' ' + subset_df['album'])
        seed_profile['text'] = vectorizer.transform([text_data])
        seed_profile['indices_to_ignore'].extend(item_indices.tolist())

    sim_text = cosine_similarity(seed_profile['text'], tfidf_matrix).flatten()
    sim_feat = cosine_similarity(seed_profile['features'].reshape(1, -1), feature_matrix).flatten()
    sim_final = weight_text * sim_text + (1.0 - weight_text) * sim_feat
    
    sim_final[[idx for idx in seed_profile['indices_to_ignore'] if idx < len(sim_final)]] = -np.inf

    indices = np.argsort(sim_final)[::-1][:top_n]
    results = [df.iloc[i].to_dict() for i in indices]
    return results, seed_profile['info']

# ===================================================================
# INTERFACE STREAMLIT
# ===================================================================

st.set_page_config(page_title="Melody RBC", layout="wide", page_icon="🎧")

st.markdown("""
<style>
    .card {
        border: 1px solid #333; border-radius: 10px; padding: 0.8rem;
        background-color: #1E1E1E; text-align: center; height: 280px;
        display: flex; flex-direction: column; justify-content: space-between;
        transition: transform 0.2s ease-in-out;
    }
    .card:hover {
        transform: scale(1.05);
        border: 1px solid #1DB954;
    }
    .card img {
        width: 100%; height: 140px; object-fit: cover; border-radius: 7px;
    }
    .card-title {
        font-weight: bold; font-size: 0.9rem; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; color: #fff;
    }
    .card-caption {
        font-size: 0.8rem; color: #aaa; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
    }
    .stButton > button {
        background-color: #1DB954; color: white; border-radius: 20px; border: none; font-weight: bold;
    }
    h1, h2, h3 { color: #1DB954; }
</style>
""", unsafe_allow_html=True)

df, feature_matrix, genres_list = load_and_process_data()

if df is None:
    st.stop()

vectorizer, tfidf_matrix = build_tfidf_matrix(df)

st.title("🎧 Melody RBC")
tab1, tab2 = st.tabs(["Recomendador Principal", "Gerador de Playlist IA ✨"])

# ======================= ABA 1: RECOMENDADOR =======================
with tab1:
    st.header("Encontre Músicas Similares")
    
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        query = st.text_input("Busque por Música, Artista ou Álbum", placeholder="Ex: Daft Punk, Get Lucky...")
    with col_search2:
        mode_map = {'Música': 'track', 'Artista': 'artist', 'Álbum': 'album'}
        mode_selection = st.selectbox("Buscar por:", mode_map.keys())
        internal_mode = mode_map[mode_selection]

    with st.expander("Filtros Avançados 🔬"):
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            year_range = st.slider("Lançado entre", int(df['year'].min()), int(df['year'].max()), (int(df['year'].min()), int(df['year'].max())))
        with col_filter2:
            selected_genres = st.multiselect("Gêneros do Artista (Top 40)", options=genres_list)
        with col_filter3:
            min_danceability = st.slider("Dançabilidade Mínima", 0.0, 1.0, 0.0)
            min_energy = st.slider("Energia Mínima", 0.0, 1.0, 0.0)

    if st.button("🔎 Buscar Recomendações", key="search_main"):
        filtered_df = df[
            (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
            (df['Danceability'] >= min_danceability) & (df['Energy'] >= min_energy)]
        if selected_genres:
            filtered_df = filtered_df[filtered_df['artist_genres'].str.contains('|'.join(selected_genres), na=False)]
        
        if query and not filtered_df.empty:
            filtered_indices = filtered_df.index
            filtered_fm = feature_matrix[filtered_indices]
            filtered_tfidf = tfidf_matrix[filtered_indices]
            df_for_rec = filtered_df.reset_index(drop=True)

            results, seed_info = get_recommendations(
                df_for_rec, filtered_fm, vectorizer, filtered_tfidf, 
                query, top_n=10, weight_text=0.6, fuzzy_threshold=0.6, mode=internal_mode)

            if results:
                st.subheader(f"Baseado em: {seed_info[internal_mode]}")
                st.markdown("---")
                cols = st.columns(5)
                for i, r in enumerate(results):
                    with cols[i % 5]:
                        # --- CORREÇÃO: Link adicionado em volta do card ---
                        spotify_url = "#"
                        track_uri = r.get('track_uri')
                        if track_uri and isinstance(track_uri, str) and 'spotify:track:' in track_uri:
                            track_id = track_uri.split(':')[-1]
                            spotify_url = f"https://open.spotify.com/track/TRACK_ID"
                        
                        st.markdown(f"""
                        <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                            <div class="card" title="{r['track']} - {r['artist']}">
                                <img src="{r.get('image_url', '')}">
                                <div>
                                    <div class="card-title">{r['track']}</div>
                                    <div class="card-caption">{r['artist']}</div>
                                </div>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Nenhum resultado encontrado com os filtros e busca atuais.")
        else:
            st.warning("Busca vazia ou nenhum dado corresponde aos filtros.")

# ======================= ABA 2: GERADOR DE PLAYLIST =======================
with tab2:
    st.header("Crie uma Playlist com base em seus Artistas Favoritos")
    
    artist_inputs = [st.text_input(f"Artista {i+1}", key=f"artist_{i}") for i in range(3)]
    
    if st.button("✨ Gerar Playlist", key="generate_playlist"):
        valid_artists = [artist for artist in artist_inputs if artist]
        if not valid_artists:
            st.error("Por favor, insira pelo menos um artista.")
        else:
            playlist = []
            seed_indices = []
            
            for artist_name in valid_artists:
                matches = get_close_matches(artist_name, df['artist'].unique(), n=1, cutoff=0.7)
                if matches:
                    matched_artist = matches[0]
                    artist_tracks = df[df['artist'] == matched_artist]
                    playlist.extend(artist_tracks.head(3).to_dict('records'))
                    seed_indices.extend(artist_tracks.index.tolist())

            if not seed_indices:
                st.warning("Nenhum dos artistas foi encontrado na base de dados.")
            else:
                combined_features = feature_matrix[seed_indices].mean(axis=0)
                subset_df = df.loc[seed_indices]
                combined_text_data = ' '.join(subset_df['track'].astype(str) + ' ' + subset_df['artist'].astype(str) + ' ' + subset_df['album'].astype(str))
                combined_text_vec = vectorizer.transform([combined_text_data])
                
                sim_text = cosine_similarity(combined_text_vec, tfidf_matrix).flatten()
                sim_feat = cosine_similarity(combined_features.reshape(1, -1), feature_matrix).flatten()
                sim_final = 0.5 * sim_text + 0.5 * sim_feat
                sim_final[seed_indices] = -np.inf
                
                rec_indices = np.argsort(sim_final)[::-1][:15 - len(playlist)]
                playlist.extend(df.iloc[i].to_dict() for i in rec_indices)
                
                random.shuffle(playlist)
                
                st.subheader(f"Sua Playlist Gerada por IA ({len(playlist)} músicas)")
                st.markdown("---")
                
                cols = st.columns(5)
                for i, track in enumerate(playlist):
                    with cols[i % 5]:
                        # --- CORREÇÃO: Link adicionado em volta do card ---
                        spotify_url = "#"
                        track_uri = track.get('track_uri')
                        if track_uri and isinstance(track_uri, str) and 'spotify:track:' in track_uri:
                            track_id = track_uri.split(':')[-1]
                            spotify_url = f"https://open.spotify.com/track/TRACK_ID"
                        
                        st.markdown(f"""
                        <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                            <div class="card" title="{track['track']} - {track['artist']}">
                                <img src="{track.get('image_url', '')}">
                                <div>
                                    <div class="card-title">{track['track']}</div>
                                    <div class="card-caption">{track['artist']}</div>
                                </div>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)