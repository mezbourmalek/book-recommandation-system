




https://github.com/user-attachments/assets/91ae1447-c624-4fe5-a878-cd87ce4d62f5


BookMatch - Semantic Book Recommender

This is a personal project I built to practice NLP and machine learning concepts by creating a book recommendation system. Instead of recommending books based on simple keyword matching, this project uses semantic search (vector embeddings) combined with text classification and emotion analysis to recommend books that actually match what the user is looking for, both in terms of content and mood.

The final result is a small web dashboard built with Gradio where you can type a description of the kind of book you want to read, filter by category and by "mood" (happy, sad, suspenseful, etc.), and get a list of book recommendations with covers and descriptions.

Motivation

I wanted a project that would let me combine a few different NLP techniques I was learning:

Vector embeddings and similarity search
Zero-shot text classification
Emotion / sentiment classification
Building a simple interactive UI around a data pipeline

Rather than doing four separate small exercises, I combined all of them into one pipeline that ends up powering a single app.

How it works

The project is organized as a pipeline made of several Jupyter notebooks, each one handling a step, and the final result is used by a Gradio app.

1. Data exploration and cleaning

The dataset used is the "7k Books with Metadata" dataset from Kaggle. In this step I:

Load the raw dataset and check for missing values (subtitle, description, categories, etc.)
Visualize missing data with a heatmap
Remove books with missing descriptions, missing page count, missing rating or missing publication year
Filter out books with very short descriptions (less than 25 words), since they don't carry enough information for the embedding model to work with
Create a "tagged description" column that combines the ISBN with the description, so that after doing the similarity search I can trace a result back to the original book
Save the cleaned data as books_cleaned.csv
2. Vector search (semantic recommendation)

This step is about turning the book descriptions into vectors so that we can do a similarity search instead of a keyword search.

The descriptions are split into a list of LangChain Document objects
Each description is embedded using the sentence-transformers/all-MiniLM-L6-v2 model through HuggingFaceEmbeddings
The embeddings are stored in a Chroma vector database
A function retrieve_semantic_recommendations takes a natural language query (for example "A book to teach children about nature") and returns the closest matching books based on the description embeddings

This is the core of the recommendation logic: it lets a user type something like "a book about friendship and loss" and get relevant results even if none of those exact words appear in the book description.

3. Text classification (Fiction vs Nonfiction)

The original dataset has hundreds of different category labels (Fiction, Juvenile Fiction, History, Poetry, etc.), which is too granular to be useful as a filter in the app. In this step I:

Map the most common categories into 4 simplified categories: Fiction, Nonfiction, Children's Fiction, Children's Nonfiction
For books that don't have a category at all, use a zero-shot classification model (facebook/bart-large-mnli) to predict whether the description sounds like Fiction or Nonfiction
Test the accuracy of the zero-shot classifier on a sample of books that already had a known category, to check how reliable it is before trusting it on the missing ones
Merge the predicted categories back into the main dataset and save it as books_with_categories.csv
4. Emotion / sentiment analysis

To be able to filter recommendations by "mood", each book description is analyzed with an emotion classification model (j-hartmann/emotion-english-distilroberta-base), which scores text on 7 emotions: anger, disgust, fear, joy, sadness, surprise and neutral.

Since a description is usually more than one sentence, each sentence is classified separately
For every book, the maximum score obtained for each emotion across all its sentences is kept (this way, a single strong emotional sentence in the description is not diluted by more neutral sentences around it)
The final emotion scores are merged with the rest of the data and saved as books_with_emotions.csv
5. Dashboard

The dashboard is built with Gradio (gradio-dashboard.py) and brings everything together:

The user types a description of what they want to read
They can optionally filter by category (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction) and by mood (Happy, Surprising, Angry, Suspenseful, Sad)
The app runs the semantic search on the query, filters the results by category if one is selected, then re-sorts the results by the chosen emotion score if a mood is selected
Results are displayed as a gallery of book covers, and clicking on a book shows its full description
Tech stack
Python
pandas / numpy for data handling
LangChain (langchain-community, langchain-text-splitters, langchain-huggingface, langchain-chroma) for the embedding and vector search pipeline
HuggingFace Transformers for zero-shot classification and emotion classification
Chroma as the vector database
Gradio for the web interface
matplotlib / seaborn / plotly for exploratory data analysis and visualizations
Dataset

The dataset used is "7k Books with Metadata" from Kaggle, downloaded using kagglehub.
Acknowledgements
This project was built as a learning exercise while following an online tutorial on building a semantic book recommender with LangChain and Gradio, then adapted and extended with my own changes along the way.






