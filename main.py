import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

# Load environment variables
load_dotenv()
PORT = int(os.getenv("PORT", 10000))  # Default to 10000 as per Render's guidelines

# Load dataset
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'cover-not-found.png', books['large_thumbnail'])

# Load and split documents
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = text_splitter.split_documents(raw_documents)

# Initialize embedding model and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "chroma_db"
db_books = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

# Function to retrieve recommendations
def retrieve_sematic_recommendations(query: str, category: str = None, tone: str = None,
                                     initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
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

    return books_recs.head(final_top_k)

# Recommendation function
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc = " ".join(description.split()[:30]) + "..."
        authors = row['authors'].split(';')
        authors_str = (
            f"{authors[0]} and {authors[1]}" if len(authors) == 2 else
            f"{', '.join(authors[:-1])}, and {authors[-1]}" if len(authors) > 2 else
            authors[0]
        )
        caption = f"{row['title']} by {authors_str} : {truncated_desc}"
        results.append((row['large_thumbnail'], caption))
    return results

# Categories and tones
categories = ["All"] + sorted(books['simple_categories'].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Gradio Interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic Book Recommender')
    with gr.Row():
        user_query = gr.Textbox(label="Enter book description", placeholder="e.g., A story about hope and redemption")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
        submit_btn = gr.Button("Recommend")

    gr.Markdown('## Recommendations')
    output = gr.Gallery(label="Top Recommendations", columns=4, rows=2)

    submit_btn.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Launch Gradio App
if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=PORT,share=True)
