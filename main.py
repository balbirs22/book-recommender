import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Load data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'].fillna('cover-not-found.png') + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'cover-not-found.png', books['large_thumbnail'])

# Load and embed descriptions
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)

def retrieve_sematic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]

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

def recommend_books(query, category, tone):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_description = " ".join(description.split()[:30]) + "..."
        authors = row['authors'].split(';')
        authors_str = ', '.join(authors[:2]) + (' and others' if len(authors) > 2 else '')
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row['large_thumbnail'], caption))
    return results

# UI definition
categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

gr_interface = gr.Interface(
    fn=recommend_books,
    inputs=[
        gr.Textbox(label="Enter a book description", placeholder="e.g. A story about forgiveness"),
        gr.Dropdown(choices=categories, label="Select a category", value="All"),
        gr.Dropdown(choices=tones, label="Select a tone", value="All")
    ],
    outputs=gr.Gallery(label="Recommended books", columns=4, rows=2),
    title="Semantic Book Recommender"
)

# FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def main():
    return gr_interface.launch(share=False, inline=True)

