import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load environment variables
load_dotenv()

# Load book data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'].fillna('') + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'] == "&fife=w800", 'cover-not-found.png', books['large_thumbnail'])

# Load and split descriptions
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = text_splitter.split_documents(raw_documents)

# Set up HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db_books = FAISS.from_documents(documents, embeddings)


# Recommendation logic
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs = books_recs.sort_values(by='joy', ascending=False)
    elif tone == "Surprising":
        books_recs = books_recs.sort_values(by='surprise', ascending=False)
    elif tone == "Angry":
        books_recs = books_recs.sort_values(by='anger', ascending=False)
    elif tone == "Suspenseful":
        books_recs = books_recs.sort_values(by='fear', ascending=False)
    elif tone == "Sad":
        books_recs = books_recs.sort_values(by='sadness', ascending=False)

    return books_recs

# Streamlit UI
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommender")

query = st.text_input("Describe the kind of book you're looking for:", placeholder="e.g. An inspiring story of self-discovery")

category = st.selectbox("Choose a category", ["All"] + sorted(books['simple_categories'].unique()))
tone = st.selectbox("Choose a tone", ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"])

if st.button("Find Recommendations") and query:
    results = retrieve_semantic_recommendations(query, category, tone)
    st.subheader("Recommended Books")
    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(row['large_thumbnail'], width=100)
        with col2:
            authors_split = row['authors'].split(';')
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row['authors']
            st.markdown(f"**{row['title']}** by *{authors_str}*")
            st.caption(" ".join(row['description'].split()[:30]) + "...")
