# Semantic Book Recommender

A semantic book recommender that suggests books based on user-provided descriptions and optional parameters like tone and category. Leveraging advanced NLP techniques—including OpenAI embeddings, sentiment analysis, and category detection from Hugging Face models—this project delivers personalized recommendations using a dataset of 7k books from Kaggle. A web dashboard built with Gradio offers a user-friendly interface for interactive recommendations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Usage](#usage)
- [Architecture & Workflow](#architecture--workflow)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The Semantic Book Recommender lets users describe the type of book they’re looking for—whether by theme, tone, or specific category—and then returns a list of recommended books. It processes and cleans the dataset, generates embeddings using the OpenAI API, and uses models from Hugging Face for sentiment analysis and categorization. The result is a dynamic recommendation system that tailors suggestions based on semantic similarities and user preferences.

## Features

- **Semantic Search:** Uses OpenAI embeddings to map book descriptions into vector space for high-quality semantic search.
- **Sentiment & Category Analysis:** Utilizes Hugging Face models to determine the tone (e.g., positive, neutral, negative) and assign categories to books.
- **Data Cleaning & Preprocessing:** Applies various techniques to clean and standardize the 7k Books dataset.
- **Interactive Dashboard:** A Gradio-based web interface allows users to input descriptions, select optional tone and category filters, and view recommendations instantly.
- **Flexible Querying:** Supports natural language queries and refines recommendations based on the provided context.

## Dataset

- **7k Books Dataset:** Sourced from Kaggle, this dataset includes metadata such as ISBN, title, author, description, and more.  
  *Note:* Ensure you have permission to use the dataset and that it complies with any licensing requirements.


## Usage

### Running the Application 

To launch the Gradio dashboard, run:

``` bash
python gradio-dashboard.py
```

This command starts a local web server. Open your browser and navigate to the provided URL (typically http://localhost:7860) to interact with the recommender.

### Example Workflow
1. Input Query: Type in a description of the book you are looking for (e.g., "A book to teach children about nature").
2. Optional Filters: Optionally, select a tone (e.g., "Inspirational", "Serious") and category (e.g., "Fiction", "Nonfiction").
3. Get Recommendations: Click the "Recommend" button to view a list of matching books.

## Architecture & Workflow
1. **Data Preprocessing:**

- Clean and preprocess the 7k Books dataset.
- Extract key fields and prepare text data for embeddings.

2. **Embedding Generation:**

- Use the OpenAI API to generate embeddings for each book's description.
- Store embeddings in a vector store for efficient similarity search.

3. **Sentiment & Category Analysis:**

- Apply Hugging Face models to analyze sentiment and infer categories for books with unspecified categories.
  
4. **Recommendation Engine:**

- Convert the user query into an embedding.
- Perform a similarity search to retrieve the most relevant books.
- Optionally filter recommendations based on tone and category.
  
5. **Web Interface:**

- A Gradio-based dashboard provides an interactive user experience.

## Contributing
Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please open an issue or submit a pull request. For major changes, discuss them via an issue first to ensure alignment with the project’s goals.

## Acknowledgements
- **OpenAI:** For powering the semantic embeddings.
- **Hugging Face:** For providing pre-trained models for sentiment analysis and categorization.
- **Kaggle:** For the 7k Books dataset.
- **Gradio:** For the intuitive web interface framework.
- Thanks to all contributors and the open-source community for their support and resources.
