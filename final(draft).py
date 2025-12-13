

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

# Set your email for Entrez
Entrez.email = "omar.atwaa16@gmail.com"

def fetch_article_details(pmid: str):
    """Fetch details for a single article by PMID"""
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="xml",
            retmode="xml"
        )

        record = Entrez.read(handle)
        handle.close()

        article = record['PubmedArticle'][0]
        medline = article['MedlineCitation']
        article_data = medline['Article']

        # Abstract handling
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
    """Search PubMed and return PMIDs"""
    # Add year filter to query if specified
    if year and year != "All Years":
        query = f"{query} AND {year}[pdat]"

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    record = Entrez.read(handle)
    handle.close()

    pmids = record.get("IdList", [])
    return pmids

def fetch_multiple_articles(pmids, delay: float = 0.4):
    """Fetch multiple articles with progress bar"""
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

# =========================
# TOKENIZATION & INDEX
# =========================
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9\-]+\b")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_RE.findall(text)]

def build_inverted_index(documents: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build or update an inverted index from a dict of documents"""
    # Load existing index if exists
    if os.path.exists("inverted_index.pkl"):
        with open("inverted_index.pkl", "rb") as f:
            inverted_index = pickle.load(f)
    else:
        inverted_index = defaultdict(set)

    # Update index
    for doc_id, text in documents.items():
        for token in tokenize(text):
            inverted_index[token].add(doc_id)

    # Save updated index
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    return dict(inverted_index)

# -------------------------------
# Basic Metrics: Precision & Recall & F1
# -------------------------------

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@k = (# relevant docs in top k) / k
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    rel_retrieved = sum(1 for d in retrieved_k if d in relevant)
    return rel_retrieved / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@k = (# relevant docs in top k) / (# relevant docs)
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    rel_retrieved = sum(1 for d in retrieved_k if d in relevant)
    return rel_retrieved / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """
    F1 = 2 * P * R / (P + R)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Streamlit App
st.set_page_config(page_title="PubMed Article Search", page_icon="🔬", layout="wide")

st.title("🔬 PubMed Article Search")
st.markdown("Search and retrieve scientific articles from PubMed")

# Create tabs
tab1, tab2 = st.tabs(["📄 Single Article by PMID", "🔍 Search Multiple Articles"])

# Tab 1: Single Article
with tab1:
    st.header("Fetch Single Article by PMID")

    pmid_input = st.text_input("Enter PubMed ID (PMID):", placeholder="e.g., 31311655")

    if st.button("Fetch Article", key="fetch_single"):
        if pmid_input:
            with st.spinner("Fetching article..."):
                article = fetch_article_details(pmid_input)

                if article:
                    st.success("Article retrieved successfully!")

                    # Display article details
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

                    if article['authors']:
                        with st.expander("All Authors"):
                            st.write(", ".join(article['authors']))
        else:
            st.warning("Please enter a PMID")

# Tab 2: Multiple Articles Search
with tab2:
    st.header("Search Multiple Articles")

    query_input = st.text_input("Enter search query:", placeholder="e.g., cancer immunotherapy")

    col1, col2 = st.columns(2)

    with col1:
        year_filter = st.selectbox(
            "Filter by Year:",
            ["All Years"] + [str(year) for year in range(2024, 1999, -1)]
        )

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
                        # Convert to DataFrame
                        df = pd.DataFrame(articles)

                        # Format authors column
                        if 'authors' in df.columns:
                            df['authors_str'] = df['authors'].apply(
                                lambda x: '; '.join(x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else str(x)
                            )

                        # Format keywords column
                        if 'keywords' in df.columns:
                            df['keywords_str'] = df['keywords'].apply(
                                lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x)
                            )

                        # Extract year from pub_date
                        df['year'] = df['pub_date'].apply(lambda x: x.get('Year', 'N/A') if isinstance(x, dict) else 'N/A')

                        st.subheader(f"Results: {len(articles)} articles")

                        # Display articles
                        for idx, article in enumerate(articles):
                            with st.expander(f"{idx + 1}. {article['title']}", expanded=(idx == 0)):
                                col1, col2 = st.columns([2, 1])

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
                                    st.markdown(f"**Authors:** {', '.join(article['authors'][:5])}{'...' if len(article['authors']) > 5 else ''}")

                                st.markdown("**Abstract:**")
                                st.write(article['abstract'] if article['abstract'] else "No abstract available")

                        # Download CSV
                        st.markdown("---")
                        csv_df = df[['pmid', 'title', 'abstract', 'authors_str', 'journal', 'year', 'keywords_str']].copy()
                        csv_df.columns = ['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Year', 'Keywords']

                        csv = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"pubmed_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No articles found for your query")
        else:
            st.warning("Please enter a search query")

st.markdown("---")
st.caption("Data retrieved from PubMed using NCBI Entrez API")