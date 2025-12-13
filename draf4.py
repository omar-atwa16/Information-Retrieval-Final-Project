import streamlit as st
import time
from datetime import datetime
import pandas as pd
from Bio import Entrez
import re
from collections import defaultdict
from typing import Dict, Set, List
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# download nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# ENTrez CONFIG
# =========================
Entrez.email = "omar.atwaa16@gmail.com"

# =========================
# TEXT PREPROCESSOR CLASS (exactly as you sent)
# =========================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter = PorterStemmer()
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text)

    def normalize(self, tokens):
        return [t.lower() for t in tokens if t.isalpha()]

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens, algorithm='porter'):
        if algorithm == 'porter':
            return [self.porter.stem(t) for t in tokens]
        elif algorithm == 'snowball':
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def process_text(self, text):
        tokens = self.tokenize(text)          # fixed typo here
        normalized = self.normalize(tokens)
        filtered = self.remove_stopwords(normalized)
        stemmed = self.stem(filtered)
        lemmatized = self.lemmatize(stemmed)
        return lemmatized

preprocessor = TextPreprocessor()

# =========================
# INVERTED INDEX
# =========================
def build_inverted_index(documents: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build or update an inverted index from a dict of documents"""
    if os.path.exists("inverted_index.pkl"):
        with open("inverted_index.pkl", "rb") as f:
            inverted_index = pickle.load(f)
    else:
        inverted_index = defaultdict(set)

    for doc_id, text in documents.items():
        tokens = preprocessor.process_text(text)
        for token in tokens:
            inverted_index[token].add(doc_id)

    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    return dict(inverted_index)

# =========================
# FETCH ARTICLES (same as before)
# =========================
def fetch_article_details(pmid: str):
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        article = record['PubmedArticle'][0]
        medline = article['MedlineCitation']
        article_data = medline['Article']

        # Abstract
        abstract_text = ""
        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
            abstract_parts = article_data['Abstract']['AbstractText']
            if isinstance(abstract_parts, list):
                abstract_text = " ".join(str(x) for x in abstract_parts)
            else:
                abstract_text = str(abstract_parts)

        # Authors
        authors_list = []
        for author in article_data.get('AuthorList', []):
            last = author.get('LastName', '')
            initials = author.get('Initials', '')
            full = f"{last} {initials}".strip()
            if full:
                authors_list.append(full)

        # Keywords
        keywordsList = []
        if 'KeywordList' in medline:
            for kw_list in medline['KeywordList']:
                for kw in kw_list:
                    keywordsList.append(str(kw))

        details = {
            'pmid': pmid,
            'title': article_data.get('ArticleTitle', ''),
            'abstract': abstract_text,
            'authors': authors_list,
            'journal': article_data.get('Journal', {}).get('Title', ''),
            'pub_date': article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}),
            'keywords': keywordsList
        }
        return details
    except Exception as e:
        st.error(f"Error fetching PMID {pmid}: {e}")
        return None

def search_pubmed(query: str, max_results: int = 10, year: str = None):
    if year and year != "All Years":
        query = f"{query} AND {year}[pdat]"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])

def fetch_multiple_articles(pmids, delay: float = 0.4):
    articles = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, pmid in enumerate(pmids):
        status_text.text(f"Fetching article {i + 1}/{len(pmids)}: PMID {pmid}")
        details = fetch_article_details(pmid)
        if details:
            articles.append(details)
        progress_bar.progress((i + 1) / len(pmids))
        time.sleep(delay)
    progress_bar.empty()
    status_text.empty()
    return articles

# ======================
# EVALUATION METRICS
# ======================
def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k

def recall_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

# =========================
# STREAMLIT APP (same layout as before)
# =========================
st.set_page_config(page_title="PubMed Article Search + Inverted Index", page_icon="🔬", layout="wide")
st.title("🔬 PubMed Article Search + Inverted Index")
st.markdown("Search and retrieve scientific articles from PubMed and build a persistent inverted index")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 ID Search",
    "📄 Query Search",
    "📊 Evalcuation",
    "Inverted index Search"
])

# --- TAB 1 ---
with tab1:
    st.header("Fetch Single Article by PMID")
    pmid_input = st.text_input("Enter PubMed ID (PMID):", placeholder="e.g., 31311655")
    if st.button("Fetch Article", key="fetch_single"):
        if pmid_input:
            with st.spinner("Fetching article..."):
                article = fetch_article_details(pmid_input)
                if article:
                    st.success("Article retrieved successfully!")
                    st.subheader(article['title'])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**PMID:** {article['pmid']}")
                        st.markdown(f"**Journal:** {article['journal']}")
                        pub_year = article['pub_date'].get('Year', 'N/A')
                        pub_month = article['pub_date'].get('Month', '')
                        st.markdown(f"**Publication Date:** {pub_month} {pub_year}")
                    with col2:
                        if article['authors']:
                            st.markdown(f"**Authors:** {', '.join(article['authors'][:3])}{'...' if len(article['authors']) > 3 else ''}")
                        if article['keywords']:
                            st.markdown(f"**Keywords:** {', '.join(article['keywords'][:5])}")
                    st.markdown("---")
                    st.subheader("Abstract")
                    st.write(article['abstract'] if article['abstract'] else "No abstract available")
        else:
            st.warning("Please enter a PMID")

# --- TAB 2 ---
with tab2:
    st.header("Search Multiple Articles")
    query_input = st.text_input("Enter search query:", placeholder="e.g., cancer immunotherapy")
    col1, col2 = st.columns(2)
    with col1:
        year_filter = st.selectbox("Filter by Year:", ["All Years"] + [str(year) for year in range(2024, 1999, -1)])
    with col2:
        max_results = st.slider("Number of results:", min_value=1, max_value=100, value=10)

    if st.button("Search Articles", key="search_multiple"):
        if query_input:
            with st.spinner("Searching PubMed..."):
                pmids = search_pubmed(query_input, max_results, year_filter)
                if pmids:
                    st.success(f"Found {len(pmids)} articles. Fetching details...")
                    articles = fetch_multiple_articles(pmids)
                    if articles:
                        df = pd.DataFrame(articles)
                        df['authors_str'] = df['authors'].apply(lambda x: '; '.join(x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else str(x))
                        df['keywords_str'] = df['keywords'].apply(lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x))
                        df['year'] = df['pub_date'].apply(lambda x: x.get('Year', 'N/A') if isinstance(x, dict) else 'N/A')
                        st.subheader(f"Results: {len(articles)} articles")
                        # Display articles
                        for idx, article in enumerate(articles):
                            with st.expander(f"{idx + 1}. {article['title']}", expanded=(idx==0)):
                                col1, col2 = st.columns([2,1])
                                with col1:
                                    st.markdown(f"**PMID:** {article['pmid']}")
                                    st.markdown(f"**Journal:** {article['journal']}")
                                    pub_year = article['pub_date'].get('Year', 'N/A')
                                    pub_month = article['pub_date'].get('Month', '')
                                    st.markdown(f"**Date:** {pub_month} {pub_year}")
                                with col2:
                                    if article['keywords']:
                                        st.markdown(f"**Keywords:** {', '.join(article['keywords'][:3])}")
                                if article['authors']:
                                    st.markdown(f"**Authors:** {', '.join(article['authors'][:5])}{'...' if len(article['authors'])>5 else ''}")
                                st.markdown("**Abstract:**")
                                st.write(article['abstract'] if article['abstract'] else "No abstract available")

                        # --- INVERTED INDEX UPDATE ---
                        doc_dict = {art['pmid']: art['abstract'] for art in articles if art['abstract']}
                        inverted_index = build_inverted_index(doc_dict)
                        st.success(f"Inverted index updated! Total tokens: {len(inverted_index)}")

                        st.subheader("Sample tokens in index")
                        for token in list(inverted_index.keys())[:10]:
                            st.write(f"**{token}**: {list(inverted_index[token])[:5]}")

                        # Download CSV
                        st.markdown("---")
                        csv_df = df[['pmid','title','abstract','authors_str','journal','year','keywords_str']].copy()
                        csv_df.columns = ['PMID','Title','Abstract','Authors','Journal','Year','Keywords']
                        csv = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Results as CSV", data=csv,
                                           file_name=f"pubmed_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                           mime="text/csv")
                else:
                    st.warning("No articles found for your query")

st.markdown("---")
st.caption("Data retrieved from PubMed using NCBI Entrez API")


# ======================
# TAB 3 — EVALUATION
# ======================
with tab3:
    if "retrieved_docs" not in st.session_state:
        st.warning("Nothing to evaluate yet.")
    else:
        k = st.slider("k", 1, len(st.session_state["retrieved_docs"]), 5)
        retrieved = st.session_state["retrieved_docs"]
        relevant = set(retrieved)  # assumption

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        f = f1(p, r)

        st.metric("Precision", round(p, 3))
        st.metric("Recall", round(r, 3))
        st.metric("F1-score", round(f, 3))

# ---------------------------
# TAB 4 — FILTER CACHED ARTICLES USING INVERTED INDEX
# ---------------------------

with tab4:
    if "article_cache" not in st.session_state or not st.session_state["article_cache"]:
        st.info("No articles in cache yet. Run a search first in Tab 1.")
    else:
        # input token to filter
        search_token = st.text_input("Enter a keyword to filter cached articles:").lower().strip()

        if st.button("Filter Cache"):
            if not search_token:
                st.warning("Please enter a token to filter.")
            else:
                # load inverted index
                if os.path.exists("inverted_index.pkl"):
                    with open("inverted_index.pkl", "rb") as f:
                        inverted_index = pickle.load(f)
                else:
                    inverted_index = {}

                # get matching pmids from index
                matching_pmids = inverted_index.get(search_token, set())

                # get articles from cache
                filtered_articles = [
                    art for pmid, art in st.session_state["article_cache"].items()
                    if pmid in matching_pmids
                ]

                if filtered_articles:
                    st.success(f"{len(filtered_articles)} article(s) match '{search_token}'")
                    for art in filtered_articles:
                        st.subheader(art["title"])
                        st.write(art["abstract"])
                        st.divider()
                else:
                    st.info(f"No cached articles contain the token '{search_token}'")
