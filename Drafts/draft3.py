import streamlit as st
import time
from datetime import datetime
import pandas as pd
from Bio import Entrez
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os

# ======================
# CONFIG
# ======================
Entrez.email = "omar.atwaa16@gmail.com"
CURRENT_YEAR = datetime.now().year

nltk.download("stopwords", quiet=True)

# ======================
# PREPROCESSOR (UNCHANGED)
# ======================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def process(self, text):
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

preprocessor = TextPreprocessor()

# ======================
# INVERTED INDEX (PERSISTENT)
# ======================
def load_index():
    if os.path.exists("inverted_index.pkl"):
        with open("inverted_index.pkl", "rb") as f:
            return pickle.load(f)
    return defaultdict(set)

def save_index(index):
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(index, f)

def update_inverted_index(docs):
    index = load_index()
    for doc_id, text in docs.items():
        for token in preprocessor.process(text):
            index[token].add(doc_id)
    save_index(index)
    return index

# ======================
# PUBMED FUNCTIONS
# ======================
def search_pubmed(query, year_from, year_to, max_results):
    query = f"{query} AND ({year_from}[PDAT] : {year_to}[PDAT])"
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_article(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    record = Entrez.read(handle)
    handle.close()

    article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]
    abstract = ""
    if "Abstract" in article:
        abstract = " ".join(map(str, article["Abstract"]["AbstractText"]))

    return {
        "pmid": pmid,
        "title": article.get("ArticleTitle", ""),
        "abstract": abstract
    }

# ======================
# EVALUATION METRICS
# ======================
def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k

def recall_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

# ======================
# STREAMLIT UI
# ======================
st.set_page_config("PubMed IR", layout="wide")
st.title("🔬 PubMed Information Retrieval System")

tab1, tab2, tab3 = st.tabs([
    "🔍 Search Articles",
    "📄 Article Viewer",
    "📊 Evaluation"
])

# ======================
# TAB 1 — ORIGINAL SEARCH (WITH SLIDER)
# ======================
with tab1:
    query = st.text_input("Search query")
    max_results = st.slider("Number of articles", 1, 20, 5)

    year_range = st.slider(
        "Publication year range",
        min_value=1980,
        max_value=CURRENT_YEAR,
        value=(2000, CURRENT_YEAR)
    )

    if st.button("Search"):
        pmids = search_pubmed(query, year_range[0], year_range[1], max_results)

        articles = []
        for pmid in pmids:
            articles.append(fetch_article(pmid))
            time.sleep(0.3)

        if articles:
            df = pd.DataFrame(articles)
            st.dataframe(df[["pmid", "title"]])

            docs = {a["pmid"]: a["abstract"] for a in articles if a["abstract"]}
            update_inverted_index(docs)

            st.session_state["last_query"] = query
            st.session_state["retrieved_docs"] = [a["pmid"] for a in articles]

# ======================
# TAB 2 — ORIGINAL DISPLAY
# ======================
with tab2:
    if "retrieved_docs" not in st.session_state:
        st.info("Run a search first.")
    else:
        for pmid in st.session_state["retrieved_docs"]:
            art = fetch_article(pmid)
            st.subheader(art["title"])
            st.write(art["abstract"])
            st.divider()

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
