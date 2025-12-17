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
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# =========================
# ENTrez CONFIG
# =========================
Entrez.email = "omar.atwaa16@gmail.com"

# =========================
# TEXT PREPROCESSOR CLASS
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
        tokens = self.tokenize(text)
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
# FETCH ARTICLES
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
                article = fetch_article_details(pmid_input)
                if article:
                    # Add to cache
                    st.session_state["article_cache"][pmid_input] = article
                    # Update inverted index
                    doc_dict = {pmid_input: article['abstract']}
                    build_inverted_index(doc_dict)
                    
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
                    articles = fetch_multiple_articles(pmids)
                    if articles:
                        # Add to cache
                        for article in articles:
                            st.session_state["article_cache"][article['pmid']] = article
                        
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
                    title = st.session_state["article_cache"][pmid]['title']
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
    if not os.path.exists("inverted_index.pkl"):
        st.warning("No inverted index found. Run a search in 'Query Search' tab to build the index first.")
    else:
        # Load inverted index
        with open("inverted_index.pkl", "rb") as f:
            inverted_index = pickle.load(f)
        
        st.success(f"Inverted index loaded! Total tokens: {len(inverted_index)}")
        
        # input token to search
        search_token = st.text_input("Enter a keyword to search articles:", placeholder="e.g., cancer")
        
        if st.button("Search"):
            if not search_token:
                st.warning("Please enter a token to search.")
            else:
                # Process the search token the same way
                processed_tokens = preprocessor.process_text(search_token)
                
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
                                with st.expander(f"{idx}. {art['title']}", expanded=(idx==1)):
                                    col1, col2 = st.columns([2,1])
                                    with col1:
                                        st.markdown(f"**PMID:** {art['pmid']}")
                                        st.markdown(f"**Journal:** {art['journal']}")
                                        pub_year = art['pub_date'].get('Year', 'N/A')
                                        pub_month = art['pub_date'].get('Month', '')
                                        st.markdown(f"**Date:** {pub_month} {pub_year}")
                                    with col2:
                                        if art['keywords']:
                                            st.markdown(f"**Keywords:** {', '.join(art['keywords'][:3])}")
                                    if art['authors']:
                                        st.markdown(f"**Authors:** {', '.join(art['authors'][:5])}{'...' if len(art['authors'])>5 else ''}")
                                    st.markdown("**Abstract:**")
                                    st.write(art['abstract'] if art['abstract'] else "No abstract available")
                            
                            if len(cached_articles) < len(matching_pmids):
                                st.info(f"Showing {len(cached_articles)} cached articles. {len(matching_pmids) - len(cached_articles)} articles not in cache.")
                        else:
                            st.warning("No cached articles found for these PMIDs. Run searches in 'Query Search' tab to cache articles.")
                    else:
                        st.info(f"Token '{search_token_processed}' not found in the inverted index.")

st.markdown("---")
st.caption("Data retrieved from PubMed using NCBI Entrez API")