import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load books data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'cover-not-found.png', books['large_thumbnail'])

# Load and process text data
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)

# Recommendation logic
def retrieve_sematic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category and category != "All":
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]

    tone_sorting = {
        "Happy": 'joy',
        "Surprising": 'surprise',
        "Angry": 'anger',
        "Suspenseful": 'fear',
        "Sad": 'sadness'
    }

    if tone in tone_sorting:
        books_recs = books_recs.sort_values(by=tone_sorting[tone], ascending=False)

    return books_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_description = " ".join(description.split()[:30]) + "..."
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']
        caption = f"**{row['title']}** by {authors_str}\n\n{truncated_description}"
        results.append((row['large_thumbnail'], caption))
    return results

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📚 Semantic Book Recommender")

query = st.text_input("Enter a description of a book:", placeholder="e.g. A story about forgiveness")
categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Select a category:", categories, index=0)
with col2:
    tone = st.selectbox("Select a tone:", tones, index=0)

if st.button("🔍 Find Recommendations") and query:
    results = recommend_books(query, category, tone)
    for thumbnail, caption in results:
        with st.container():
            col_img, col_txt = st.columns([1, 3])
            with col_img:
                st.image(thumbnail, use_column_width=True)
            with col_txt:
                st.markdown(caption)
