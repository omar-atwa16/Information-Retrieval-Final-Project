# Standard library
import os
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Set, List

# Third-party libraries
import nltk
import pandas as pd
from Bio import Entrez
import streamlit as st

# NLTK-specific imports
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloading/Updating nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Entrez email configuration
Entrez.email = 'omar.atwaa16@gmail.com'

# Preprocessing class
class TextPreprocessor:
    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text)

    def normalize(self, tokens):
        return [token.lower() for token in tokens if token.isalpha()]

    def removeStopwords(self, tokens):
        return [token for token in tokens if token not in self.stopWords]

    def stem(self, tokens, algorithm='porter'):
        if algorithm == 'porter':
            return [self.porter.stem(token) for token in tokens]
        return tokens

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def processText(self, text):
        tokens = self.tokenize(text)
        normalized = self.normalize(tokens)
        filtered = self.removeStopwords(normalized)
        stemmed = self.stem(filtered)
        lemmatized = self.lemmatize(stemmed)
        return lemmatized

# creating a text preprocessor instance
preprocessor = TextPreprocessor()

# Building an inverted index
def invertedIDX(docs: Dict[str, str]) -> Dict[str, Set[str]]:
    if os.path.exists('invertedIDX.pkl'):
        with open("invertedIDX.pkl", "rb") as file:
            invertedIDX = pickle.load(file)
    else:
        invertedIDX = defaultdict(set)

    for docID, text in docs.items():
        tokens = preprocessor.processText(text)
        for token in tokens:
            invertedIDX[token].add(docID)

    with open("invertedIDX.pkl", "wb") as file:
        pickle.dump(invertedIDX, file)

    return dict(invertedIDX)

# Getting details [Abstracts, Authors, ...]
def fetchArticleDetails(pmid: str):
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        article = record['PubmedArticle'][0]
        medline = article['MedlineCitation']
        articleData = medline['Article']

        abstractText = ""
        if 'Abstract' in articleData and 'AbstractText' in articleData['Abstract']:
            abstractParts = articleData['Abstract']['AbstractText']
            if isinstance(abstractParts, list):
                abstractText = " ".join(str(x) for x in abstractParts)
            else:
                abstractText = str(abstractParts)

        authors = []
        for author in articleData.get('AuthorList', []):
            first = author.get('Initials', '')
            last = author.get('LastName', '')
            full = f"{first} {last}".strip()
            if full:
                authors.append(full)

            details = {
            'ID': pmid,
            'Title': articleData.get('ArticleTitle', ''),
            'Abstract': abstractText,
            'Authors': authors,
            'Journal': articleData.get('Journal', {}).get('Title', ''),
            'Publish Date': articleData.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}),
        }
        return details
    except Exception as e:
        st.error(f"Error fetching PMID{pmid}: {e}")
        return None

# Querying PubMed
def searchPubMed(query: str, maxResults: int, year: str = None):
    if year and year != "All Years":
        query = f"{query} AND {year}"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=maxResults, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])

# getting multiple articles at once
def fetchMultiArticles(pmIDs, delay: float = 0.3):
    articles = []
    progressBar = st.progress(0)
    statusText = st.empty()
    for i, pmid in enumerate(pmIDs):
        statusText.text(f"Fetching article {i + 1}/{len(pmIDs)}: PMID {pmid}")
        details = fetchArticleDetails(pmid)
        if details:
            articles.append(details)
        progressBar.progress((i+1) / len(pmIDs))
        time.sleep(delay)
    progressBar.empty()
    statusText.empty()
    return articles

# PubMed does NOT provide truth ground so will be using these metrics assuming all data retrieved are relevant 
def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k if k > 0 else 0

def recall_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant) if len(relevant) > 0 else 0

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

# =========================
# INITIALIZE SESSION STATE
# =========================
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []  # List of dicts: {query, timestamp, retrieved_docs, relevant_docs}
if "article_cache" not in st.session_state:
    st.session_state["article_cache"] = {}

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="PubMed Article Search + Inverted Index", page_icon="🔬", layout="wide")
st.title("🔬 PubMed Article Search + Inverted Index")
st.markdown("Search and retrieve scientific articles from PubMed and build a persistent inverted index")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 ID Search",
    "📄 Query Search",
    "📊 Evaluation",
    "Inverted Index Search"
])

# --- TAB 1 ---
with tab1:
    st.header("Fetch Single Article by PMID")
    pmid_input = st.text_input("Enter PubMed ID (PMID):", placeholder="e.g., 31311655")
    if st.button("Fetch Article", key="fetch_single"):
        if pmid_input:
            with st.spinner("Fetching article..."):
                article = fetchArticleDetails(pmid_input)
                if article:
                    # Add to cache
                    st.session_state["article_cache"][pmid_input] = article
                    # Update inverted index
                    doc_dict = {pmid_input: article['Abstract']}
                    invertedIDX(doc_dict)
                    
                    st.success("Article retrieved successfully!")
                    st.subheader(article['Title'])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**PMID:** {article['ID']}")
                        st.markdown(f"**Journal:** {article['Journal']}")
                        pub_year = article['Publish Date'].get('Year', 'N/A')
                        pub_month = article['Publish Date'].get('Month', '')
                        st.markdown(f"**Publication Date:** {pub_month} {pub_year}")
                    with col2:
                        if article['Authors']:
                            st.markdown(f"**Authors:** {', '.join(article['Authors'][:3])}{'...' if len(article['Authors']) > 3 else ''}")
                    st.markdown("---")
                    st.subheader("Abstract")
                    st.write(article['Abstract'] if article['Abstract'] else "No abstract available")
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
                pmids = searchPubMed(query_input, max_results, year_filter)
                if pmids:
                    # Store search in history
                    search_entry = {
                        "query": query_input,
                        "year": year_filter,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "retrieved_docs": pmids,
                        "relevant_docs": set(pmids)  # For demo, assume all retrieved are relevant
                    }
                    st.session_state["search_history"].append(search_entry)
                    
                    st.success(f"Found {len(pmids)} articles. Fetching details...")
                    articles = fetchMultiArticles(pmids)
                    if articles:
                        # Add to cache
                        for article in articles:
                            st.session_state["article_cache"][article['ID']] = article
                        
                        df = pd.DataFrame(articles)
                        df['authors_str'] = df['Authors'].apply(lambda x: '; '.join(x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else str(x))
                        df['year'] = df['Publish Date'].apply(lambda x: x.get('Year', 'N/A') if isinstance(x, dict) else 'N/A')
                        st.subheader(f"Results: {len(articles)} articles")
                        # Display articles
                        for idx, article in enumerate(articles):
                            with st.expander(f"{idx + 1}. {article['Title']}", expanded=(idx==0)):
                                col1, col2 = st.columns([2,1])
                                with col1:
                                    st.markdown(f"**PMID:** {article['ID']}")
                                    st.markdown(f"**Journal:** {article['Journal']}")
                                    pub_year = article['Publish Date'].get('Year', 'N/A')
                                    pub_month = article['Publish Date'].get('Month', '')
                                    st.markdown(f"**Date:** {pub_month} {pub_year}")
                                if article['Authors']:
                                    st.markdown(f"**Authors:** {', '.join(article['Authors'][:5])}{'...' if len(article['Authors'])>5 else ''}")
                                st.markdown("**Abstract:**")
                                st.write(article['Abstract'] if article['Abstract'] else "No abstract available")

                        # --- INVERTED INDEX UPDATE ---
                        doc_dict = {art['ID']: art['Abstract'] for art in articles if art['Abstract']}
                        inverted_index = invertedIDX(doc_dict)
                        st.success(f"Inverted index updated! Total tokens: {len(inverted_index)}")

                        st.subheader("Sample tokens in index")
                        for token in list(inverted_index.keys())[:10]:
                            st.write(f"**{token}**: {list(inverted_index[token])[:5]}")

                        # Download CSV
                        st.markdown("---")
                        csv_df = df[['ID','Title','Abstract','authors_str','Journal','year']].copy()
                        csv_df.columns = ['PMID','Title','Abstract','Authors','Journal','Year']
                        csv = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Results as CSV", data=csv,
                                        file_name=f"pubmed_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv")
                else:
                    st.warning("No articles found for your query")

# ======================
# TAB 3 — EVALUATION (PER QUERY)
# ======================
with tab3:
    st.header("Evaluation Metrics - Per Query")
    
    if not st.session_state["search_history"]:
        st.warning("No search results to evaluate yet. Run a search in the 'Query Search' tab first.")
    else:
        st.info(f"Total queries evaluated: {len(st.session_state['search_history'])}")
        
        # Single k value slider for all queries
        max_k_overall = max(len(entry['retrieved_docs']) for entry in st.session_state["search_history"])
        k = st.slider("k value (applies to all queries):", 1, max_k_overall, min(5, max_k_overall))
        
        st.markdown("---")
        
        # Build table data
        table_data = []
        for idx, search_entry in enumerate(st.session_state["search_history"], 1):
            retrieved = search_entry['retrieved_docs']
            relevant = search_entry['relevant_docs']
            
            # Calculate metrics with the current k
            k_actual = min(k, len(retrieved))
            p = precision_at_k(retrieved, relevant, k_actual)
            r = recall_at_k(retrieved, relevant, k_actual)
            f = f1(p, r)
            
            table_data.append({
                'Query #': idx,
                'Query': search_entry['query'],
                'Year': search_entry['year'],
                'Timestamp': search_entry['timestamp'],
                'Docs Retrieved': len(retrieved),
                'Precision@k': round(p, 3),
                'Recall@k': round(r, 3),
                'F1-score': round(f, 3)
            })
        
        # Display as dataframe
        df_metrics = pd.DataFrame(table_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Optional: Show details for selected query
        st.subheader("Query Details")
        selected_query = st.selectbox("Select a query to view retrieved documents:", 
                                    [f"Query {i+1}: {entry['query']}" for i, entry in enumerate(st.session_state["search_history"])])
        
        if selected_query:
            query_idx = int(selected_query.split(":")[0].split()[1]) - 1
            selected_entry = st.session_state["search_history"][query_idx]
            
            st.markdown(f"**Query:** {selected_entry['query']}")
            st.markdown(f"**Retrieved Documents (Top 10):**")
            for i, pmid in enumerate(selected_entry['retrieved_docs'][:10], 1):
                if pmid in st.session_state["article_cache"]:
                    title = st.session_state["article_cache"][pmid]['Title']
                    st.text(f"{i}. PMID: {pmid} - {title[:80]}...")
                else:
                    st.text(f"{i}. PMID: {pmid}")
            if len(selected_entry['retrieved_docs']) > 10:
                st.text(f"... and {len(selected_entry['retrieved_docs']) - 10} more")

# ---------------------------
# TAB 4 — SEARCH USING INVERTED INDEX
# ---------------------------
with tab4:
    st.header("Search Articles by Token (Using Inverted Index)")
    
    # Check if inverted index exists
    if not os.path.exists("invertedIDX.pkl"):
        st.warning("No inverted index found. Run a search in 'Query Search' tab to build the index first.")
    else:
        # Load inverted index
        with open("invertedIDX.pkl", "rb") as f:
            inverted_index = pickle.load(f)
        
        st.success(f"Inverted index loaded! Total tokens: {len(inverted_index)}")
        
        # input token to search
        search_token = st.text_input("Enter a keyword to search articles:", placeholder="e.g., cancer")
        
        if st.button("Search"):
            if not search_token:
                st.warning("Please enter a token to search.")
            else:
                # Process the search token the same way
                processed_tokens = preprocessor.processText(search_token)
                
                if not processed_tokens:
                    st.warning("Search token was filtered out during preprocessing.")
                else:
                    search_token_processed = processed_tokens[0]
                    
                    # get matching pmids from index
                    matching_pmids = inverted_index.get(search_token_processed, set())

                    if matching_pmids:
                        st.success(f"{len(matching_pmids)} article(s) found for '{search_token}' (processed as '{search_token_processed}')")
                        st.markdown(f"**Matching PMIDs:** {', '.join(matching_pmids)}")
                        
                        st.markdown("---")
                        st.subheader("Top 10 Articles (from cache)")
                        
                        # Filter only cached articles
                        cached_articles = [
                            st.session_state["article_cache"][pmid] 
                            for pmid in matching_pmids 
                            if pmid in st.session_state["article_cache"]
                        ][:10]
                        
                        if cached_articles:
                            for idx, art in enumerate(cached_articles, 1):
                                with st.expander(f"{idx}. {art['Title']}", expanded=(idx==1)):
                                    col1, col2 = st.columns([2,1])
                                    with col1:
                                        st.markdown(f"**PMID:** {art['ID']}")
                                        st.markdown(f"**Journal:** {art['Journal']}")
                                        pub_year = art['Publish Date'].get('Year', 'N/A')
                                        pub_month = art['Publish Date'].get('Month', '')
                                        st.markdown(f"**Date:** {pub_month} {pub_year}")
                                    if art['Authors']:
                                        st.markdown(f"**Authors:** {', '.join(art['Authors'][:5])}{'...' if len(art['Authors'])>5 else ''}")
                                    st.markdown("**Abstract:**")
                                    st.write(art['Abstract'] if art['Abstract'] else "No abstract available")
                            
                            if len(cached_articles) < len(matching_pmids):
                                st.info(f"Showing {len(cached_articles)} cached articles. {len(matching_pmids) - len(cached_articles)} articles not in cache.")
                        else:
                            st.warning("No cached articles found for these PMIDs. Run searches in 'Query Search' tab to cache articles.")
                    else:
                        st.info(f"Token '{search_token_processed}' not found in the inverted index.")

st.markdown("---")
st.caption("Data retrieved from PubMed using NCBI Entrez API")