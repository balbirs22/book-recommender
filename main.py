import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file (for local use)
load_dotenv()

# Load and prepare book data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'cover-not-found.png', books['large_thumbnail'])

# Load and split documents
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator='\n', length_function=len)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings and Chroma vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)

# Recommendation logic
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
        books_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    return books_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_sematic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_description = " ".join(description.split()[:30]) + "..."
        authors = row['authors'].split(';')
        if len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = row['authors']
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row['large_thumbnail'], caption))
    return results

# Gradio UI setup
categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic Book Recommender')
    with gr.Row():
        user_query = gr.Textbox(label="Enter a description of a book:", placeholder="e.g. A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown('# Recommendations')
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Launch app using port from .env or default 7860
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    dashboard.launch(server_name="0.0.0.0", server_port=port)
