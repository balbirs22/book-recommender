import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables
load_dotenv()
PORT = int(os.getenv("PORT", 10000))

# Load data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'].fillna('cover-not-found.png') + "&fife=w800"

# Load and split documents
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = splitter.split_documents(raw_documents)

# Optional: filter overly long chunks (Render memory safety)
documents = [doc for doc in documents if len(doc.page_content) <= 2000]

# Create or load persisted vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
else:
    db_books = Chroma.from_documents(documents, embedding_model, persist_directory=PERSIST_DIR)
    db_books.persist()

def retrieve_sematic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50,
                                     final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    tone_map = {
        "Happy": 'joy',
        "Surprising": 'surprise',
        "Angry": 'anger',
        "Suspenseful": 'fear',
        "Sad": 'sadness'
    }

    if tone in tone_map:
        books_recs = books_recs.sort_values(by=tone_map[tone], ascending=False)

    return books_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        truncated_desc = " ".join(row['description'].split()[:30]) + "..."
        authors = row['authors'].split(';')
        author_str = (
            f"{authors[0]} and {authors[1]}" if len(authors) == 2
            else f"{', '.join(authors[:-1])}, and {authors[-1]}" if len(authors) > 2
            else authors[0]
        )
        caption = f"{row['title']} by {author_str} : {truncated_desc}"
        results.append((row['large_thumbnail'], caption))
    return results

# UI setup
categories = ["All"] + sorted(books['simple_categories'].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# 📚 Semantic Book Recommender')
    with gr.Row():
        user_query = gr.Textbox(label="Enter book description", placeholder="e.g. A story about resilience")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
        submit_button = gr.Button("Find Recommendations")
    gr.Markdown('# 🧠 Recommended Books')
    output = gr.Gallery(label="Recommendations", columns=4, rows=2)
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Launch app
if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=PORT)
