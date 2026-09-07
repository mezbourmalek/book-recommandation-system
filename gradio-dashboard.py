import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr


load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=10000,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)


def retrieve_sementic_rec(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
):
    recs = db_books.similarity_search_with_score(
        query,
        k=initial_top_k
    )

    results = []

    for rank, (doc, score) in enumerate(recs):

        content = doc.page_content.strip('"')

        isbn = content.split(":")[0]

        try:
            isbn = int(isbn)
        except (ValueError, TypeError):
            continue

        results.append({
            "isbn13": isbn,
            "score": score,
            "rank": rank
        })

    if not results:
        return pd.DataFrame()

    semantic_results = pd.DataFrame(results)

    books_recs = books.merge(
        semantic_results,
        on="isbn13",
        how="inner"
    )

    books_recs = books_recs.sort_values(
        by="rank"
    )

    if category != "All":
        books_recs = books_recs[
            books_recs["simple_categories"] == category
        ]

    if tone == "Happy":

        books_recs = books_recs.sort_values(
            by="joy",
            ascending=False
        )

    elif tone == "Surprising":

        books_recs = books_recs.sort_values(
            by="surprise",
            ascending=False
        )

    elif tone == "Angry":

        books_recs = books_recs.sort_values(
            by="anger",
            ascending=False
        )

    elif tone == "Suspenseful":

        books_recs = books_recs.sort_values(
            by="fear",
            ascending=False
        )

    elif tone == "Sad":

        books_recs = books_recs.sort_values(
            by="sadness",
            ascending=False
        )

    books_recs = books_recs.head(final_top_k)

    return books_recs


def format_authors(authors):

    if pd.isna(authors):
        return "Unknown author"

    authors = str(authors).strip()

    if not authors:
        return "Unknown author"

    authors_split = [
        author.strip()
        for author in authors.split(";")
        if author.strip()
    ]

    if not authors_split:
        return "Unknown author"

    if len(authors_split) == 1:
        return authors_split[0]

    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"

    return (
        f"{', '.join(authors_split[:-1])} "
        f"and {authors_split[-1]}"
    )


def recommend_books(
    query: str,
    category: str,
    tone: str
):

    if not query or not query.strip():
        return [], []

    recommendations = retrieve_sementic_rec(
        query=query,
        category=category,
        tone=tone
    )

    if recommendations.empty:
        return [], []

    gallery_data = []
    full_data = []

    for _, row in recommendations.iterrows():

        authors_str = format_authors(
            row["authors"]
        )

        title = (
            str(row["title"])
            if pd.notna(row["title"])
            else "Unknown title"
        )

        caption = f"{title}\nby {authors_str}"

        image = row["large_thumbnail"]

        if pd.isna(image) or not image:
            image = "cover-not-found.jpg"

        description = (
            str(row["description"]).strip()
            if "description" in row and pd.notna(row["description"])
            else "No description available."
        )

        gallery_data.append(
            (image, caption)
        )

        full_data.append(
            {
                "image": image,
                "title": title,
                "authors": authors_str,
                "description": description
            }
        )

    return gallery_data, full_data


def on_select_book(evt: gr.SelectData, current_results):

    if not current_results or evt.index is None or evt.index >= len(current_results):
        return gr.update(value=None), gr.update(value=""), gr.update(visible=False)

    book = current_results[evt.index]

    detail_html = (
        f"<h3>{book['title']}</h3>"
        f"<p class='book-authors'>by {book['authors']}</p>"
        f"<p class='book-description'>{book['description']}</p>"
    )

    return gr.update(value=book["image"]), gr.update(value=detail_html), gr.update(visible=True)


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


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#f7f8fc",
    body_background_fill_dark="#f7f8fc",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e8eaf0",
    block_border_color_dark="#e8eaf0",
    block_label_text_color="#363946",
    block_label_text_color_dark="#363946",
    body_text_color="#20232d",
    body_text_color_dark="#20232d",
    body_text_color_subdued="#5b5f6e",
    body_text_color_subdued_dark="#5b5f6e",
    input_background_fill="#fafbfe",
    input_background_fill_dark="#fafbfe",
    input_border_color="#e1e4ec",
    input_border_color_dark="#e1e4ec",
    button_primary_background_fill="#6c4df6",
    button_primary_background_fill_dark="#6c4df6",
    button_primary_background_fill_hover="#5b3ce6",
    button_primary_background_fill_hover_dark="#5b3ce6",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)


custom_css = """
:root, .dark {
    --body-text-color: #20232d !important;
    --background-fill-primary: #f7f8fc !important;
    --block-background-fill: #ffffff !important;
    --border-color-primary: #e8eaf0 !important;
}

body, .dark body {
    background: #f7f8fc !important;
}

.gradio-container, .dark .gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding: 35px 45px 60px !important;
    background: #f7f8fc !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#title {
    text-align: center;
    margin-top: 15px !important;
    margin-bottom: 4px !important;
}

#title h1 {
    font-size: 46px !important;
    font-weight: 800 !important;
    letter-spacing: -1.8px !important;
    color: #171923 !important;
}

#subtitle {
    text-align: center;
    color: #73778a !important;
    font-size: 17px !important;
    margin-top: 0 !important;
    margin-bottom: 32px !important;
}

.search-panel, .dark .search-panel {
    background: #ffffff !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 22px !important;
    padding: 25px !important;
    margin-bottom: 38px !important;
    box-shadow: 0 8px 30px rgba(25, 30, 50, 0.06) !important;
}

.search-panel *, .dark .search-panel * {
    background-color: transparent;
}

#search-input {
    margin-bottom: 18px !important;
}

#search-input textarea, .dark #search-input textarea {
    background: #fafbfe !important;
    border: 1px solid #e1e4ec !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    color: #20232d !important;
    padding: 15px !important;
}

#search-input textarea:focus {
    border-color: #7c5cff !important;
    box-shadow: 0 0 0 3px rgba(124, 92, 255, 0.12) !important;
}

.search-panel label span, .dark .search-panel label span {
    color: #363946 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

#category-dropdown, .dark #category-dropdown,
#mood-dropdown, .dark #mood-dropdown {
    background: #fafbfe !important;
}

#category-dropdown .wrap-inner, .dark #category-dropdown .wrap-inner,
#mood-dropdown .wrap-inner, .dark #mood-dropdown .wrap-inner {
    border-radius: 13px !important;
    border: 1px solid #e1e4ec !important;
    background: #fafbfe !important;
}

#category-dropdown input, .dark #category-dropdown input,
#mood-dropdown input, .dark #mood-dropdown input,
#category-dropdown span, .dark #category-dropdown span,
#mood-dropdown span, .dark #mood-dropdown span {
    color: #20232d !important;
    background: transparent !important;
    -webkit-text-fill-color: #20232d !important;
    opacity: 1 !important;
}

#category-dropdown ul, .dark #category-dropdown ul,
#mood-dropdown ul, .dark #mood-dropdown ul {
    background: #ffffff !important;
    color: #20232d !important;
}

#category-dropdown li, .dark #category-dropdown li,
#mood-dropdown li, .dark #mood-dropdown li {
    color: #20232d !important;
}

#search-button {
    height: 52px !important;
    min-width: 150px !important;
    border-radius: 13px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    background: #6c4df6 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 6px 15px rgba(108, 77, 246, 0.22) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

#search-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 9px 20px rgba(108, 77, 246, 0.28) !important;
}

#recommendation-title {
    margin-top: 5px !important;
    margin-bottom: 20px !important;
}

#recommendation-title h2 {
    color: #20222d !important;
    font-size: 25px !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px !important;
}

#book-gallery, .dark #book-gallery {
    border: none !important;
    background: transparent !important;
}

#book-gallery .grid-container {
    gap: 25px !important;
}

#book-gallery .thumbnail-item.thumbnail-lg, .dark #book-gallery .thumbnail-item.thumbnail-lg {
    background: #ffffff !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 16px !important;
    box-shadow: 0 5px 18px rgba(25, 30, 50, 0.06) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    aspect-ratio: auto !important;
    height: 300px !important;
}

#book-gallery .thumbnail-item.thumbnail-lg:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 14px 30px rgba(25, 30, 50, 0.12) !important;
}

#book-gallery .thumbnail-item.thumbnail-lg img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}

#book-gallery .caption-label, .dark #book-gallery .caption-label {
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    top: auto !important;
    max-width: 100% !important;
    width: 100% !important;
    background: rgba(255, 255, 255, 0.95) !important;
    color: #20232d !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 12px !important;
    white-space: normal !important;
    text-overflow: clip !important;
    overflow: hidden !important;
    max-height: 45% !important;
    text-align: left !important;
    border: none !important;
    border-top: 1px solid #e8eaf0 !important;
    line-height: 1.4 !important;
}

#book-detail, .dark #book-detail {
    background: #ffffff !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 18px !important;
    padding: 22px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 8px 30px rgba(25, 30, 50, 0.06) !important;
}

#book-detail-img img {
    width: 220px !important;
    height: 320px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(25, 30, 50, 0.15) !important;
}

#book-detail-text h3 {
    color: #171923 !important;
    font-size: 24px !important;
    font-weight: 750 !important;
    margin-bottom: 8px !important;
}

#book-detail-text p {
    color: #5b5f6e !important;
    font-size: 16px !important;
}

#book-detail-text .book-authors {
    font-weight: 600 !important;
    color: #6c4df6 !important;
    margin-bottom: 12px !important;
}

#book-detail-text .book-description {
    line-height: 1.6 !important;
    max-height: 260px !important;
    overflow-y: auto !important;
}

@media (max-width: 800px) {

    .gradio-container {
        padding: 25px 18px 40px !important;
    }

    #title h1 {
        font-size: 36px !important;
    }

    #subtitle {
        font-size: 15px !important;
    }

    .search-panel {
        padding: 18px !important;
    }
}
"""

with gr.Blocks(theme=theme) as dashboard:

    gr.Markdown(
        "# BookMatch",
        elem_id="title"
    )

    gr.Markdown(
        "Discover your next favorite book based on your interests and mood.",
        elem_id="subtitle"
    )

    with gr.Group(
        elem_classes="search-panel"
    ):

        user_query = gr.Textbox(
            label="What would you like to read?",
            placeholder="Try: A magical adventure with dragons...",
            lines=2,
            elem_id="search-input"
        )

        with gr.Row():

            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Category",
                value="All",
                scale=1,
                elem_id="category-dropdown"
            )

            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Mood",
                value="All",
                scale=1,
                elem_id="mood-dropdown"
            )

            submit_button = gr.Button(
                "Find books",
                variant="primary",
                scale=0,
                elem_id="search-button"
            )

    gr.Markdown(
        "## Recommended for you",
        elem_id="recommendation-title"
    )

    results_state = gr.State([])

    with gr.Row(
        elem_id="book-detail",
        visible=False
    ) as book_detail_row:

        detail_image = gr.Image(
            show_label=False,
            container=False,
            scale=0,
            elem_id="book-detail-img"
        )

        detail_text = gr.HTML(
            scale=1,
            elem_id="book-detail-text"
        )

    output = gr.Gallery(
        label="",
        columns=4,
        rows=2,
        height=650,
        object_fit="cover",
        show_label=False,
        allow_preview=False,
        elem_id="book-gallery"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=[output, results_state]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=book_detail_row
    )

    user_query.submit(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=[output, results_state]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=book_detail_row
    )

    output.select(
        fn=on_select_book,
        inputs=results_state,
        outputs=[detail_image, detail_text, book_detail_row]
    )


if __name__ == "__main__":

    dashboard.launch(
        css=custom_css
    )
