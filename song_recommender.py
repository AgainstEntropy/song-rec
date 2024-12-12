import functools
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from torch.cuda import is_available as is_cuda_available
from transformers import AutoModel

# Model and data paths
MODEL_NAME = "./models/jina-embeddings-v3"
EMBEDDINGS_PATH = "./embeddings/jina-embeddings-v3_retrieval.passage.npy"
DATA_PATH = "./data/spotify_millsongdata.csv"

@st.cache_resource
def load_model_and_data():
    """Load the model, tokenizer, embeddings, and dataset."""

    # Verify file existence
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {os.path.abspath(EMBEDDINGS_PATH)}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {os.path.abspath(DATA_PATH)}")

    # Load model and tokenizer
    try:
        with st.spinner(f"Loading model: {MODEL_NAME}"):
            model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
            if is_cuda_available():
                model.to("cuda")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

    # Load embeddings
    try:
        passage_embeddings = np.load(EMBEDDINGS_PATH)
        # st.success(f"Passage embeddings loaded: shape {passage_embeddings.shape}")
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        raise

    # Load song data
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"Song data loaded: {df.shape[0]} records")
    except Exception as e:
        st.error(f"Error loading song data: {e}")
        raise

    return model, passage_embeddings, df


def compute_similarity(
    query_texts: list[str], model, passage_embedding_shards: list[np.ndarray]
) -> np.ndarray:
    """Compute cosine similarity between user input and song embeddings."""
    query_embeddings = model.encode(query_texts, task="retrieval.query")

    # Ensure proper dimensions for similarity computation
    if query_embeddings.ndim == 2 and query_embeddings.shape[0] == 1:
        query_embeddings = query_embeddings[0]  # Remove the batch dimension (shape becomes 1D)

    def mapper(shard):
        shard_similarity = query_embeddings @ shard.T
        return shard_similarity

    def reducer(p, c):
        return np.hstack((p, c))

    # Compute similarities
    mapped = map(mapper, passage_embedding_shards)
    similarities = functools.reduce(reducer, mapped)
    return similarities


def get_top_k(similarities: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k indices and similarities for the most similar songs."""
    top_k_indices = np.argpartition(-similarities, k)[:k]
    sorted_top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
    top_k_similarities = similarities[sorted_top_k_indices]

    return sorted_top_k_indices, top_k_similarities


def retrieve_top_k_entries(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Retrieve rows of songs based on indices."""
    return df.iloc[indices]


if __name__ == "__main__":

    # Streamlit UI
    st.title("🎵 Song Recommendation System")
    st.markdown(
        """
        Welcome! Input a mood, description, or any text, and get personalized song recommendations!
        """
    )

    try:
        model, passage_embeddings, song_data = load_model_and_data()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

    num_shards = 10
    passage_embedding_shards = np.array_split(passage_embeddings, num_shards, axis=0)

    # User Input
    query = st.text_input("Enter a description of the song you're looking for:")
    num_recommendations = st.slider(
        "How many recommendations would you like?", 1, 10, 5
    )

    if st.button("Recommend Songs"):
        if query := query.strip():
            with st.spinner("Finding songs for you..."):
                try:
                    tic = time.time()
                    # Compute similarities
                    similarities = compute_similarity(
                        [query], model, passage_embedding_shards
                    )

                    # Get top-k indices
                    top_k_indices, top_k_similarities = get_top_k(
                        similarities, k=num_recommendations
                    )

                    # Retrieve recommendations
                    recommendations = retrieve_top_k_entries(song_data, top_k_indices)

                    toc = time.time()
                    st.success(f"Time taken: {toc - tic:.2f} seconds")

                    st.subheader("Your Recommendations:")
                    for i, song in enumerate(recommendations.itertuples(), 1):
                        st.markdown(
                            f"""
                            **{i}. {song.artist} - {song.song} - score: {top_k_similarities[i-1]:.3f}**
                            - Lyrics Snippet: {song.text[:200]}...
                            - [More Info]({song.link})
                            """
                        )
                except Exception as e:
                    st.error(f"Error during recommendation generation: {e}")
        else:
            st.warning("Please enter a query to get recommendations.")
