import pandas as pd
import numpy as np
import torch
import os
import gradio as gr

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print(
        f"Groq API key loaded successfully "
        f"and starts with {groq_api_key[:5]}"
    )
else:
    print("No API Key found")


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD BOOK DATASET
# =========================================================

books = pd.read_csv("books_with_emotions.csv")


# =========================================================
# HANDLE THUMBNAILS
# =========================================================

books["thumbnail"] = books["thumbnail"].fillna("")

books["large_thumbnail"] = np.where(
    books["thumbnail"] != "",
    books["thumbnail"] + "&fife=w800",
    "cover-not-found.png"
)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model
)


# =========================================================
# RETRIEVE SEMANTIC RECOMMENDATIONS
# =========================================================

def retrieve_semantic_recommendations(
    query: str,
    category: str,
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(
        query,
        k=initial_top_k
    )

    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
    ]

    book_recs = books[
        books["isbn13"].isin(books_list)
    ]

    # =====================================================
    # CATEGORY FILTER
    # =====================================================

    if category != "All":

        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    # =====================================================
    # EMOTIONAL TONE FILTER
    # =====================================================

    if tone == "Happy":

        book_recs = book_recs.sort_values(
            by="joy",
            ascending=False
        )

    elif tone == "Surprising":

        book_recs = book_recs.sort_values(
            by="surprise",
            ascending=False
        )

    elif tone == "Angry":

        book_recs = book_recs.sort_values(
            by="anger",
            ascending=False
        )

    elif tone == "Suspenseful":

        book_recs = book_recs.sort_values(
            by="fear",
            ascending=False
        )

    elif tone == "Sad":

        book_recs = book_recs.sort_values(
            by="sadness",
            ascending=False
        )

    return book_recs.head(final_top_k)


# =========================================================
# GENERATE HTML BOOK CARDS
# =========================================================

def recommend_books(
    query: str,
    category: str,
    tone: str
):

    recommendations = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )

    html = """
    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fill,minmax(250px,1fr));
        gap:25px;
        padding:20px;
    ">
    """

    for _, row in recommendations.iterrows():

        title = str(row["title"])
        authors = str(row["authors"])
        description = str(row["description"])

        # =================================================
        # SHORT DESCRIPTION
        # =================================================

        short_description = (
            " ".join(description.split()[:35]) + "..."
        )

        # =================================================
        # FORMAT AUTHORS
        # =================================================

        authors_split = authors.split(";")

        if len(authors_split) == 2:

            authors_str = (
                f"{authors_split[0]} and "
                f"{authors_split[1]}"
            )

        elif len(authors_split) > 2:

            authors_str = (
                f"{', '.join(authors_split[:-1])} "
                f"and {authors_split[-1]}"
            )

        else:

            authors_str = authors

        # =================================================
        # LINKS
        # =================================================

        google_books_link = (
            f"https://books.google.com/books?vid=ISBN"
            f"{row['isbn13']}"
        )

        amazon_link = (
            "https://www.amazon.com/s?k="
            f"{title.replace(' ', '+')}"
        )

        # =================================================
        # CARD HTML
        # =================================================

        html += f"""
        <div style="
            background:#111827;
            border-radius:20px;
            overflow:hidden;
            box-shadow:0 8px 20px rgba(0,0,0,0.4);
            transition:0.3s;
            border:1px solid #374151;
        ">

            <img
                src="{row['large_thumbnail']}"
                style="
                    width:100%;
                    height:380px;
                    object-fit:cover;
                "
            >

            <div style="padding:18px;">

                <h3 style="
                    color:white;
                    margin-bottom:10px;
                    font-size:20px;
                    line-height:1.3;
                ">
                    {title}
                </h3>

                <p style="
                    color:#9CA3AF;
                    font-size:14px;
                    margin-bottom:12px;
                ">
                    by {authors_str}
                </p>

                <p style="
                    color:#D1D5DB;
                    font-size:14px;
                    line-height:1.6;
                    margin-bottom:20px;
                ">
                    {short_description}
                </p>

                <div style="
                    display:flex;
                    gap:10px;
                    flex-wrap:wrap;
                ">

                    <a
                        href="{google_books_link}"
                        target="_blank"
                        style="
                            background:#2563EB;
                            color:white;
                            padding:10px 14px;
                            border-radius:10px;
                            text-decoration:none;
                            font-size:14px;
                            font-weight:600;
                        "
                    >
                        📖 Read Preview
                    </a>

                    <a
                        href="{amazon_link}"
                        target="_blank"
                        style="
                            background:#F59E0B;
                            color:black;
                            padding:10px 14px;
                            border-radius:10px;
                            text-decoration:none;
                            font-size:14px;
                            font-weight:600;
                        "
                    >
                        🛒 Buy Book
                    </a>

                </div>

            </div>

        </div>
        """

    html += "</div>"

    return html


# =========================================================
# DROPDOWNS
# =========================================================

categories = ["All"] + sorted(
    books["simple_categories"]
    .dropna()
    .unique()
    .tolist()
)

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad"
]


# =========================================================
# GRADIO DASHBOARD
# =========================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Semantic Book Recommender"
) as dashboard:

    gr.Markdown(
        """
        # 📚 Semantic Book Recommender
        
        Discover books using AI-powered semantic search,
        emotional tones, and category filtering.
        """
    )

    with gr.Row():

        user_query = gr.Textbox(
            label="Enter Book Description",
            placeholder=(
                "e.g., A dark fantasy story "
                "with dragons and magic..."
            ),
            scale=3
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select Category",
            value="All",
            scale=1
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select Emotional Tone",
            value="All",
            scale=1
        )

    submit_button = gr.Button(
        "🔍 Find Recommendations",
        variant="primary"
    )

    output = gr.HTML()

    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=output
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    dashboard.launch()