import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher
import re
import unicodedata
import os
import io
import joblib
import hashlib

# =====================
# Utilitários gerais
# =====================
def normalize_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def similaridade_strings(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower() if a else '', b.lower() if b else '').ratio()


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# =====================
# Normalização de metadados (sinônimos)
# =====================
ARTIST_SYNONYMS = {
    # Exemplos de padronização — adicione mais conforme a base
    'legiao urbana': 'legião urbana',
    'os paralamas do sucesso': 'paralamas do sucesso',
    'the paralamas do sucesso': 'paralamas do sucesso',
    'los hermanos arriagada': 'los hermanos',
}

GENRE_SYNONYMS = {
    'bossa-nova': 'bossa nova',
    'mpb ': 'mpb',
    'rock brasileiro': 'rock',
}


def apply_synonyms(series: pd.Series, synonyms: dict) -> pd.Series:
    def _map(v):
        if not isinstance(v, str):
            return v
        key = normalize_text(v)
        if key in synonyms:
            return synonyms[key]
        return v
    return series.apply(_map)


# =====================
# Classe MusicCBR (com cache e vizinhos rápidos)
# =====================
class MusicCBR:
    def __init__(self, df: pd.DataFrame, cache_dir: str):
        self.df = df.copy().reset_index(drop=True)

        # Garante colunas obrigatórias
        for c in ['artist', 'track', 'year', 'genre', 'lyrics']:
            if c not in self.df.columns:
                self.df[c] = ''

        # Normalização/Padronização básica de metadados
        self.df['artist'] = apply_synonyms(self.df['artist'].fillna(''), ARTIST_SYNONYMS)
        self.df['genre'] = apply_synonyms(self.df['genre'].fillna(''), GENRE_SYNONYMS)

        for c in ['artist', 'track', 'genre', 'lyrics']:
            self.df[c + '_norm'] = self.df[c].fillna('').apply(normalize_text)

        self.df['text_combined'] = (
            self.df['track_norm'] + ' ' + self.df['genre_norm'] + ' ' + self.df['lyrics_norm']
        ).fillna('')

        # Conversão de ano
        try:
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        except Exception:
            self.df['year'] = np.nan

        self.index_lookup = {i: row for i, row in self.df.iterrows()}

        # Artefatos em disco (vectorizer, matriz TF-IDF e índice de vizinhos)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        self.vectorizer, self.tfidf_matrix = self._load_or_fit_tfidf()
        self.nn_index = self._build_or_load_nn()

    def _artifact_paths(self):
        return (
            os.path.join(self.cache_dir, 'vectorizer.pkl'),
            os.path.join(self.cache_dir, 'tfidf_matrix.pkl'),
            os.path.join(self.cache_dir, 'nn_index.pkl'),
        )

    def _load_or_fit_tfidf(self):
        vec_p, mat_p, _ = self._artifact_paths()
        try:
            vectorizer = joblib.load(vec_p)
            tfidf_matrix = joblib.load(mat_p)
            # Confere dimensão
            if tfidf_matrix.shape[0] == len(self.df):
                return vectorizer, tfidf_matrix
        except Exception:
            pass
        # Treina do zero
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(self.df['text_combined'])
        joblib.dump(vectorizer, vec_p)
        joblib.dump(tfidf_matrix, mat_p)
        return vectorizer, tfidf_matrix

    def _build_or_load_nn(self):
        _, _, nn_p = self._artifact_paths()
        try:
            nn = joblib.load(nn_p)
            # Verificação superficial
            if hasattr(nn, 'kneighbors'):
                return nn
        except Exception:
            pass
        # Índice aproximado com NearestNeighbors (métrica cosseno)
        nn = NearestNeighbors(metric='cosine', algorithm='brute')
        nn.fit(self.tfidf_matrix)
        joblib.dump(nn, nn_p)
        return nn

    def find_by_title(self, title: str, threshold=0.6):
        title_norm = normalize_text(title)
        best_idx = None
        best_score = 0.0
        for idx, row in self.df.iterrows():
            s = similaridade_strings(title_norm, row['track_norm'])
            if s > best_score:
                best_score = s
                best_idx = idx
        if best_score < threshold:
            return None, best_score
        return best_idx, best_score

    def query_similar(self, source_idx=None, title=None, lyrics_snippet=None, top_n=10,
                      weight_lyrics=0.8, weight_metadata=0.2, genre_filter=None, artist_filter=None,
                      year_window=None):
        # Vetor de consulta
        if source_idx is None:
            query_track = normalize_text(title) if title else ''
            query_genre = normalize_text(genre_filter) if genre_filter else ''
            query_lyrics = normalize_text(lyrics_snippet) if lyrics_snippet else ''
            query_text = f"{query_track} {query_genre} {query_lyrics}".strip()
            if not query_text:
                raise ValueError('É necessário fornecer title ou lyrics_snippet para consulta.')
            q_vec = self.vectorizer.transform([query_text])
        else:
            q_vec = self.tfidf_matrix[source_idx]

        # Busca rápida por vizinhos
        n_neighbors = min(top_n * 50, self.tfidf_matrix.shape[0])  # busca larga, filtra depois
        distances, indices = self.nn_index.kneighbors(q_vec, n_neighbors=n_neighbors)
        sim_scores = 1 - distances.flatten()
        indices = indices.flatten()

        results = []
        for idx, sim in zip(indices, sim_scores):
            if source_idx is not None and idx == source_idx:
                continue
            # Metadados
            meta_bonus = 0.0
            if genre_filter and normalize_text(str(self.df.loc[idx, 'genre'])) == normalize_text(str(genre_filter)):
                meta_bonus += 1.0
            if artist_filter and normalize_text(str(self.df.loc[idx, 'artist'])) == normalize_text(str(artist_filter)):
                meta_bonus += 1.0

            year_bonus = 0.0
            if year_window and not np.isnan(self.df.loc[idx, 'year']):
                source_year = None
                if source_idx is not None and not np.isnan(self.df.loc[source_idx, 'year']):
                    source_year = self.df.loc[source_idx, 'year']
                # Se for consulta textual, usamos mediana dos anos como referência
                if source_year is None and year_window:
                    source_year = np.nanmedian(self.df['year']) if self.df['year'].notna().any() else None
                if source_year is not None and not np.isnan(source_year):
                    year_diff = abs(self.df.loc[idx, 'year'] - source_year)
                    # Bônus suave (gaussiano)
                    if year_diff <= year_window:
                        year_bonus = float(np.exp(-(year_diff ** 2) / (2 * (max(1, year_window / 2) ** 2))))

            final_score = (weight_lyrics * sim) + (weight_metadata * ((meta_bonus) * 0.5 + year_bonus * 0.5))
            results.append((idx, float(final_score), float(sim), float(meta_bonus), float(year_bonus)))

        # Ordena e aplica filtros finais
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        filtered = []
        for idx, final_score, sim, meta_bonus, year_bonus in results_sorted:
            if genre_filter and normalize_text(str(self.df.loc[idx, 'genre'])) != normalize_text(str(genre_filter)):
                continue
            if artist_filter and normalize_text(str(self.df.loc[idx, 'artist'])) != normalize_text(str(artist_filter)):
                continue
            filtered.append({
                'idx': idx,
                'score': final_score,
                'text_sim': sim,
                'meta_bonus': meta_bonus,
                'year_bonus': year_bonus,
                'artist': self.df.loc[idx, 'artist'] if self.df.loc[idx, 'artist'] else 'Artista desconhecido',
                'track': self.df.loc[idx, 'track'] if self.df.loc[idx, 'track'] else 'Título desconhecido',
                'genre': self.df.loc[idx, 'genre'] if self.df.loc[idx, 'genre'] else '',
                'year': int(self.df.loc[idx, 'year']) if not np.isnan(self.df.loc[idx, 'year']) else '',
                'lyrics': self.df.loc[idx, 'lyrics']
            })
            if len(filtered) >= top_n:
                break
        return filtered

    def get_case(self, idx):
        if idx in self.index_lookup:
            return self.index_lookup[idx].to_dict()
        return None


# =====================
# Configurações do Streamlit
# =====================
st.set_page_config(page_title='MelodyRBC - Recomendador', layout='wide')
st.title('MelodyRBC — Raciocínio Baseado em Casos para Músicas (versão aprimorada)')

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'tcc_ceds_music.csv')
CACHE_DIR = os.path.join(BASE_DIR, '.melody_cache')

if not os.path.exists(DATA_PATH):
    st.error('Arquivo de dados não encontrado. Coloque "tcc_ceds_music.csv" na mesma pasta deste script.')
    st.stop()

# Guardamos a assinatura do CSV para invalidar cache se a base mudar
csv_signature = file_md5(DATA_PATH)
st.sidebar.code(f"CSV MD5: {csv_signature[:10]}…", language='text')

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ['artist', 'artista', 'author', 'artist_name']:
            rename_map[c] = 'artist'
        if lc in ['track', 'title', 'faixa', 'nome', 'track_name']:
            rename_map[c] = 'track'
        if lc in ['year', 'ano', 'data', 'release_date']:
            rename_map[c] = 'year'
        if lc in ['genre', 'genero']:
            rename_map[c] = 'genre'
        if lc in ['lyrics', 'letra', 'texto']:
            rename_map[c] = 'lyrics'
    df = df.rename(columns=rename_map)
    for req in ['artist', 'track', 'genre', 'lyrics', 'year']:
        if req not in df.columns:
            df[req] = ''
        df[req] = df[req].fillna('')
    return df

# Recarrega modelos quando assinatura do CSV mudar
@st.cache_resource(show_spinner=True)
def load_cbr_model(sig: str):
    df = load_data(DATA_PATH)
    return MusicCBR(df, CACHE_DIR)

cbr = load_cbr_model(csv_signature)
df = cbr.df

# =====================
# Sidebar (UI aprimorada)
# =====================
st.sidebar.subheader('Configurações')
top_n = st.sidebar.slider('Número de recomendações', 1, 30, 8)
weight_lyrics = st.sidebar.slider('Peso: Conteúdo (letra/título)', 0.0, 1.0, 0.9)
weight_metadata = 1.0 - weight_lyrics
year_window = st.sidebar.slider('Janela temporal (± anos, 0 desliga)', 0, 50, 10)

# Gêneros disponíveis (normalizados/limpos)
unique_genres = sorted({g for g in df['genre'].dropna().unique() if str(g).strip()})
genre_filter_choice = st.sidebar.selectbox('Filtrar por gênero (opcional)', options=[''] + unique_genres, index=0)
artist_filter_text = st.sidebar.text_input('Filtrar por artista (opcional)')

st.sidebar.markdown('---')
st.sidebar.write('💾 Artefatos de cache: vectorizer, TF-IDF e índice de vizinhos.')

# =====================
# Modos de uso
# =====================
mode = st.radio('Escolha o modo', ['Buscar por música (título)', 'Buscar por trecho da letra', 'Explorar artista', 'Detalhes por índice'])

# ---------- Buscar por título ----------
if mode == 'Buscar por música (título)':
    query = st.text_input('Digite o título (ou parte do título) da música:')
    if st.button('Buscar similares', type='primary'):
        if not query.strip():
            st.warning('Digite o título de uma música.')
        else:
            found_idx, title_score = cbr.find_by_title(query, threshold=0.3)
            if found_idx is None:
                st.info('Nenhum título suficientemente próximo — buscando por conteúdo (título/letra/gênero).')
                results = cbr.query_similar(title=query, top_n=top_n, weight_lyrics=weight_lyrics,
                                            weight_metadata=weight_metadata, genre_filter=genre_filter_choice or None,
                                            artist_filter=artist_filter_text or None,
                                            year_window=(year_window if year_window > 0 else None))
                base_case = None
            else:
                st.success(f'Título encontrado com similaridade {title_score:.2f}.')
                base_case = cbr.get_case(found_idx)
                results = cbr.query_similar(source_idx=found_idx, top_n=top_n, weight_lyrics=weight_lyrics,
                                            weight_metadata=weight_metadata, genre_filter=genre_filter_choice or None,
                                            artist_filter=artist_filter_text or None,
                                            year_window=(year_window if year_window > 0 else None))

            if base_case:
                st.subheader('Caso base (consulta)')
                st.markdown(
                    f"**{base_case['track']}** — {base_case['artist']} ({base_case['year']})  \nGênero: {base_case['genre']}"
                )
                with st.expander('Letra / texto do caso base'):
                    txt = str(base_case['lyrics'])
                    st.write(txt[:2000] + ('...' if len(txt) > 2000 else ''))

            st.subheader('Recomendações (casos recuperados)')
            if results:
                rows = []
                for r in results:
                    # Transparência do ranking
                    text_part = weight_lyrics * r['text_sim']
                    meta_part = weight_metadata * ((r['meta_bonus']) * 0.5 + r['year_bonus'] * 0.5)
                    with st.container(border=True):
                        st.markdown(
                            f"**{r['track']}** — {r['artist']} ({r['year']})  \n"
                            f"Gênero: {r['genre']}  \n"
                            f"Score final: {r['score']:.3f} — TextoSim: {r['text_sim']:.3f}"
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write('Contribuição do conteúdo (letra/título):')
                            st.progress(min(1.0, float(text_part)))
                        with col2:
                            st.write('Contribuição de metadados (gênero/artista/ano):')
                            st.progress(min(1.0, float(meta_part)))

                        with st.expander('Trecho da letra / texto'):
                            txt = str(r['lyrics'])
                            st.write(txt[:1500] + ('...' if len(txt) > 1500 else ''))

                    rows.append({k: r[k] for k in ['artist', 'track', 'genre', 'year', 'score', 'text_sim', 'meta_bonus', 'year_bonus']})

                # Exportar CSV
                if rows:
                    out_df = pd.DataFrame(rows)
                    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Exportar recomendações (CSV)', data=csv_bytes, file_name='melodyrbc_recomendacoes.csv', mime='text/csv')
            else:
                st.info('Nenhuma música encontrada.')

# ---------- Buscar por trecho da letra ----------
elif mode == 'Buscar por trecho da letra':
    snippet = st.text_area('Cole um trecho da letra ou descreva o que procura:')
    if st.button('Buscar por trecho', type='primary'):
        if not snippet.strip():
            st.warning('Insira um trecho da letra ou descrição.')
        else:
            results = cbr.query_similar(
                lyrics_snippet=snippet,
                top_n=top_n,
                weight_lyrics=1.0,
                weight_metadata=0.0,
                genre_filter=genre_filter_choice or None,
                artist_filter=artist_filter_text or None,
                year_window=(year_window if year_window > 0 else None)
            )

            st.subheader('Recomendações (casos recuperados)')
            mostrou = False
            rows = []
            if results:
                for r in results:
                    if r['text_sim'] < 0.15:
                        continue
                    text_part = 1.0 * r['text_sim']
                    meta_part = 0.0
                    with st.container(border=True):
                        st.markdown(
                            f"**{r['track']}** — {r['artist']} ({r['year']})  \n"
                            f"Gênero: {r['genre']}  \n"
                            f"Score final: {r['score']:.3f} — TextoSim: {r['text_sim']:.3f}"
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write('Contribuição do conteúdo:')
                            st.progress(min(1.0, float(text_part)))
                        with col2:
                            st.write('Contribuição de metadados:')
                            st.progress(min(1.0, float(meta_part)))
                        with st.expander('Trecho da letra / texto'):
                            txt = str(r['lyrics'])
                            st.write(txt[:1500] + ('...' if len(txt) > 1500 else ''))
                    rows.append({k: r[k] for k in ['artist', 'track', 'genre', 'year', 'score', 'text_sim']})
                    mostrou = True

            if rows:
                out_df = pd.DataFrame(rows)
                csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                st.download_button('Exportar recomendações (CSV)', data=csv_bytes, file_name='melodyrbc_recomendacoes_letra.csv', mime='text/csv')

            if not mostrou:
                st.info('Nenhuma música suficientemente similar encontrada. Tente inserir um trecho maior ou diferente.')

# ---------- Explorar artista ----------
elif mode == 'Explorar artista':
    artist_q = st.text_input('Digite o nome do artista:')
    if st.button('Buscar artista', type='primary'):
        if not artist_q.strip():
            st.warning('Digite o nome do artista.')
        else:
            artist_norm = normalize_text(artist_q)
            matches = df[df['artist'].fillna('').apply(lambda x: normalize_text(x) == artist_norm)]
            if matches.empty:
                df_tmp = df.copy()
                df_tmp['artist_sim'] = df_tmp['artist'].fillna('').apply(lambda x: similaridade_strings(artist_q, x))
                top_artist = df_tmp.sort_values('artist_sim', ascending=False).iloc[0]
                if top_artist['artist_sim'] > 0.5 and top_artist['artist']:
                    st.info(f'Nenhuma correspondência exata. Mostrando músicas do artista mais parecido: **{top_artist["artist"]}**')
                    top_artist_norm = normalize_text(top_artist['artist'])
                    matches = df[df['artist'].fillna('').apply(lambda x: similaridade_strings(top_artist_norm, normalize_text(x)) > 0.85)]
                    if not matches.empty:
                        for _, row in matches.iterrows():
                            st.markdown(
                                f"**{row['track'] if row['track'] else 'Título desconhecido'}** — "
                                f"{row['artist']} ({row['year'] if not pd.isna(row['year']) else ''}) — "
                                f"Gênero: {row['genre'] if row['genre'] else ''}"
                            )
                    else:
                        st.info('Nenhuma música encontrada para o artista mais parecido.')
                else:
                    st.info('Nenhuma correspondência exata ou artista similar encontrada.')
            else:
                st.success(f'Foram encontradas {len(matches)} músicas de {artist_q}')
                for _, row in matches.iterrows():
                    st.markdown(
                        f"**{row['track'] if row['track'] else 'Título desconhecido'}** — "
                        f"{row['artist']} ({row['year'] if not pd.isna(row['year']) else ''}) — "
                        f"Gênero: {row['genre'] if row['genre'] else ''}"
                    )

# ---------- Detalhes por índice ----------
elif mode == 'Detalhes por índice':
    idx = st.number_input('Digite o índice (idx) da música (0 a %d):' % (len(df) - 1), min_value=0, max_value=len(df) - 1, value=0, step=1)
    if st.button('Mostrar detalhes', type='primary'):
        case = cbr.get_case(int(idx))
        if case:
            st.json(case)
        else:
            st.warning('Índice inválido.')

# =====================
# Rodapé
# =====================
st.sidebar.markdown('---')
st.sidebar.write('Backend: MusicCBR — TF-IDF + índice de vizinhos (cosine).')
st.sidebar.write('Para bases grandes, considere FAISS/Annoy. Adicione sinônimos em ARTIST_SYNONYMS/GENRE_SYNONYMS.')
