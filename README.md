# 🎧 Melody RBC - Sistema de Recomendação de Músicas


**Melody RBC** é uma aplicação web interativa, construída com **Streamlit**, que utiliza uma abordagem de IA de **Raciocínio Baseado em Casos (RBC)** — também conhecida como *Filtragem Baseada em Conteúdo* — para gerar recomendações musicais personalizadas.

A aplicação analisa características textuais (gênero, artista) e sonoras (dançabilidade, energia, etc.) de uma música, artista ou álbum para encontrar outras músicas com uma "vibe" similar na base de dados.

---

##  Principais Funcionalidades

- **Recomendação Inteligente:** Busque por música, artista ou álbum e receba recomendações baseadas em conteúdo.  
- **Filtros Avançados:** Refine sua busca por ano de lançamento, gênero, nível de dançabilidade e energia.  
- **Gerador de Playlist IA:** Insira até 3 dos seus artistas favoritos e a IA criará uma playlist mesclando músicas deles com novas descobertas de artistas similares.  
- **Interface Interativa:** Uma interface limpa e amigável construída com Streamlit.  

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**  
- **Streamlit** (para a interface web)  
- **Pandas** (para manipulação de dados)  
- **Scikit-learn** (para a lógica de Machine Learning - TF-IDF e Similaridade)  
- **Numpy** (para computação numérica)  

---

## ⚙️ Instalação e Configuração

Siga os passos abaixo para rodar o projeto localmente.

### 1. Clone o Repositório
\`\`\`sh
git clone https://github.com/seu-usuario/MelodyRBC.git
cd MelodyRBC
\`\`\`

### 2. Obtenha a Base de Dados
Este projeto utiliza a base de dados **top_10000_1960-now**. 

- Após o download, extraia o arquivo compactado (\`.zip\`).  
- Mova o arquivo **top_10000_1960-now** para a raiz do projeto, na mesma pasta onde está o arquivo **app.py**.  

---

## ▶️ Como Executar

Com o ambiente virtual ativado e as dependências instaladas, execute o seguinte comando no terminal:

\`\`\`sh
streamlit run app.py
\`\`\`

A aplicação será iniciada e aberta automaticamente no seu navegador.

---

## 📂 Estrutura do Projeto
\`\`\`
MelodyRBC/
├── .streamlit/  
│   └── config.toml      # (Opcional) Configurações de tema do Streamlit  
├── app.py               # Código principal da aplicação Streamlit  
├── top_10000_1960-now   # Arquivo da base de dados (deve ser baixado)  
└── README.md            # Documentação do projeto  
\`\`\`

## **🤝 Colaboradores**
| [<img src="https://avatars.githubusercontent.com/u/142179999?v=4" width="115">](https://github.com/Luiz-Przygoda) | [<img src="https://avatars.githubusercontent.com/u/113839563?v=4" width="115">](https://github.com/Wyllye) | [<img src="https://avatars.githubusercontent.com/u/125486974?v=4" width="115">](https://github.com/mariaglx) | [<img src="https://avatars.githubusercontent.com/u/75136675?v=4" width="115">](https://github.com/marcobgh)|
|:--------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| **Luiz-Przygoda**                                                              | **Wyllye**                                                               | **Mariaglx**                                                           | **Marcobgh**                                                              |
