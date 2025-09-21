import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
import unicodedata
import os

# =====================
# Funções utilitárias
# =====================
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def similaridade_strings(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower() if a else '', b.lower() if b else '').ratio()

# =====================
# Classe MusicCBR
# =====================
class MusicCBR:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        for c in ['artist', 'track', 'year', 'genre', 'lyrics']:
            if c not in self.df.columns:
                self.df[c] = ""
        for c in ['artist', 'track', 'genre', 'lyrics']:
            self.df[c + '_norm'] = self.df[c].fillna('').apply(normalize_text)
        self.df['text_combined'] = (self.df['track_norm'] + ' ' + self.df['genre_norm'] + ' ' + self.df['lyrics_norm']).fillna('')
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text_combined'])
        try:
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        except Exception:
            self.df['year'] = np.nan
        self.index_lookup = {i: row for i, row in self.df.iterrows()}

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
        if source_idx is None:
            # Sempre monte o texto igual ao treino
            query_track = normalize_text(title) if title else ''
            query_genre = normalize_text(genre_filter) if genre_filter else ''
            query_lyrics = normalize_text(lyrics_snippet) if lyrics_snippet else ''
            query_text = f"{query_track} {query_genre} {query_lyrics}".strip()
            if not query_text:
                raise ValueError('É necessário fornecer title ou lyrics_snippet para consulta.')
            q_vec = self.vectorizer.transform([query_text])
            sim_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        else:
            q_vec = self.tfidf_matrix[source_idx]
            sim_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        results = []
        for idx, sim in enumerate(sim_scores):
            if source_idx is not None and idx == source_idx:
                continue
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
                if source_year is not None:
                    year_diff = abs(self.df.loc[idx, 'year'] - source_year)
                    if year_diff <= year_window:
                        year_bonus = max(0.0, (year_window - year_diff) / year_window)
            final_score = (weight_lyrics * sim) + (weight_metadata * ((meta_bonus) * 0.5 + year_bonus * 0.5))
            results.append((idx, final_score, sim, meta_bonus, year_bonus))

        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        filtered = []
        for idx, final_score, sim, meta_bonus, year_bonus in results_sorted:
            if genre_filter and normalize_text(str(self.df.loc[idx, 'genre'])) != normalize_text(str(genre_filter)):
                continue
            if artist_filter and normalize_text(str(self.df.loc[idx, 'artist'])) != normalize_text(str(artist_filter)):
                continue
            filtered.append({
                'idx': idx,
                'score': float(final_score),
                'text_sim': float(sim),
                'meta_bonus': float(meta_bonus),
                'year_bonus': float(year_bonus),
                'artist': self.df.loc[idx, 'artist'] if self.df.loc[idx, 'artist'] else "Artista desconhecido",
                'track': self.df.loc[idx, 'track'] if self.df.loc[idx, 'track'] else "Título desconhecido",
                'genre': self.df.loc[idx, 'genre'] if self.df.loc[idx, 'genre'] else "",
                'year': self.df.loc[idx, 'year'] if not np.isnan(self.df.loc[idx, 'year']) else "",
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
st.title('MelodyRBC — Raciocínio Baseado em Casos para Músicas')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tcc_ceds_music.csv')
if not os.path.exists(DATA_PATH):
    st.error('Arquivo de dados não encontrado. Coloque "tcc_ceds_music.csv" na mesma pasta deste script.')
    st.stop()

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
    # Garante que todas as colunas obrigatórias existem
    for req in ['artist', 'track', 'genre', 'lyrics', 'year']:
        if req not in df.columns:
            df[req] = ''
        df[req] = df[req].fillna('')
    return df

df = load_data(DATA_PATH)

# =====================
# Sidebar
# =====================
st.sidebar.subheader('Configurações (foco no backend)')
top_n = st.sidebar.slider('Número de recomendações', 1, 20, 6)
weight_lyrics = st.sidebar.slider('Peso: Lyrics/Texto', 0.0, 1.0, 0.95)
weight_metadata = 1.0 - weight_lyrics
year_window = st.sidebar.slider('Janela de ano (± anos) (0 desliga)', 0, 50, 10)
genre_filter = st.sidebar.text_input('Filtrar por gênero (opcional)')
artist_filter = st.sidebar.text_input('Filtrar por artista (opcional)')
st.sidebar.markdown('---')
st.sidebar.write('Dicas: o sistema usa TF-IDF sobre título+gênero+letra. Ajuste o peso para priorizar letra ou metadados.')

# =====================
# Inicialização do CBR
# =====================
with st.spinner('Inicializando backend RBC (treinando TF-IDF)...'):
    cbr = MusicCBR(df)

# =====================
# Modos de busca
# =====================
mode = st.radio('Escolha o modo', ['Buscar por música (título)', 'Buscar por trecho da letra', 'Explorar artista', 'Detalhes por índice'])

# ---------- Buscar por título ----------
if mode == 'Buscar por música (título)':
    query = st.text_input('Digite o título (ou parte do título) da música:')
    if st.button('Buscar similares'):
        if not query.strip():
            st.warning('Digite o título de uma música.')
        else:
            found_idx, title_score = cbr.find_by_title(query, threshold=0.3)
            if found_idx is None:
                st.info('Nenhum título similar suficientemente próximo encontrado — faremos busca por trecho/título combinados.')
                results = cbr.query_similar(title=query, top_n=top_n, weight_lyrics=weight_lyrics,
                                            weight_metadata=weight_metadata, genre_filter=genre_filter or None,
                                            artist_filter=artist_filter or None,
                                            year_window=(year_window if year_window>0 else None))
            else:
                st.success(f'Título encontrado com similaridade {title_score:.2f}:')
                st.write(cbr.get_case(found_idx))
                results = cbr.query_similar(source_idx=found_idx, top_n=top_n, weight_lyrics=weight_lyrics,
                                            weight_metadata=weight_metadata, genre_filter=genre_filter or None,
                                            artist_filter=artist_filter or None,
                                            year_window=(year_window if year_window>0 else None))
            st.subheader('Caso base (consulta)')
            if found_idx is not None:
                base_case = cbr.get_case(found_idx)
                st.markdown(
                    f"**{base_case['track']}** — {base_case['artist']} ({base_case['year']})  \n"
                    f"Gênero: {base_case['genre']}"
                )
                with st.expander('Letra / texto do caso base'):
                    st.write(str(base_case['lyrics'])[:1000] + ('...' if len(str(base_case['lyrics']))>1000 else ''))
            st.subheader('Recomendações (casos recuperados)')
            if results:
                for r in results:
                    motivos = []
                    if r['text_sim'] > 0.7:
                        motivos.append("Letra/título muito similar")
                    if r['meta_bonus'] > 0.5:
                        motivos.append("Metadados similares (gênero/artista)")
                    if r['year_bonus'] > 0.3:
                        motivos.append("Ano próximo")
                    st.markdown(
                        f"**{r['track']}** — {r['artist']} ({r['year']})  \n"
                        f"Gênero: {r['genre']}  \n"
                        f"Score final: {r['score']:.3f} — TextoSim: {r['text_sim']:.3f}  \n"
                        f"Motivos: {', '.join(motivos) if motivos else 'Similaridade geral'}"
                    )
                    with st.expander('Trecho da letra / texto'):
                        st.write(str(r['lyrics'])[:1000] + ('...' if len(str(r['lyrics']))>1000 else ''))
            else:
                st.info('Nenhuma música encontrada.')

# ---------- Buscar por trecho da letra ----------
elif mode == 'Buscar por trecho da letra':
    snippet = st.text_area('Cole um trecho da letra ou descreva o que procura:')
    if st.button('Buscar por trecho'):
        if not snippet.strip():
            st.warning('Insira um trecho da letra ou descrição.')
        else:
            results = cbr.query_similar(
                lyrics_snippet=snippet,
                top_n=top_n,
                weight_lyrics=1.0,  # Peso máximo para letra
                weight_metadata=0.0,
                genre_filter=genre_filter or None,
                artist_filter=artist_filter or None,
                year_window=(year_window if year_window > 0 else None)
            )
            st.subheader('Recomendações (casos recuperados)')
            mostrou = False
            if results:
                for r in results:
                    if r['text_sim'] < 0.15:  # threshold mais baixo
                        continue
                    motivos = []
                    if r['text_sim'] > 0.7:
                        motivos.append("Letra/título muito similar")
                    if r['meta_bonus'] > 0.5:
                        motivos.append("Metadados similares (gênero/artista)")
                    if r['year_bonus'] > 0.3:
                        motivos.append("Ano próximo")
                    st.markdown(
                        f"**{r['track']}** — {r['artist']} ({r['year']})  \n"
                        f"Gênero: {r['genre']}  \n"
                        f"Score final: {r['score']:.3f} — TextoSim: {r['text_sim']:.3f}  \n"
                        f"Motivos: {', '.join(motivos) if motivos else 'Similaridade geral'}"
                    )
                    with st.expander('Trecho da letra / texto'):
                        st.write(str(r['lyrics'])[:1000] + ('...' if len(str(r['lyrics']))>1000 else ''))
                    mostrou = True

            if not mostrou:
                st.info('Nenhuma música suficientemente similar encontrada. Tente inserir um trecho maior ou diferente.')

# ---------- Explorar artista ----------
elif mode == 'Explorar artista':
    artist_q = st.text_input('Digite o nome do artista:')
    if st.button('Buscar artista'):
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
                    # Mostra todas as músicas com nome de artista similar ao top_artist (usando nomes normalizados)
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
                    st.info('Nenhuma correspondência exata ou artista similar encontrado.')
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
    idx = st.number_input('Digite o índice (idx) da música (0 a %d):' % (len(df)-1), min_value=0, max_value=len(df)-1, value=0, step=1)
    if st.button('Mostrar detalhes'):
        case = cbr.get_case(int(idx))
        if case:
            st.json(case)
        else:
            st.warning('Índice inválido.')

# =====================
# Sidebar info
# =====================
st.sidebar.markdown('---')
st.sidebar.write('Backend: MusicCBR — TF-IDF + RBC retrieval.')
st.sidebar.write('Observação: este protótipo concentra-se na lógica de recuperação de casos; para produção considere salvar o vectorizer e a matriz TF-IDF para rapidez.')
