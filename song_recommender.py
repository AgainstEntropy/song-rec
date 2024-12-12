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
        song_data = pd.read_csv(DATA_PATH)
        st.success(f"Song data loaded: {song_data.shape[0]} records")
    except Exception as e:
        st.error(f"Error loading song data: {e}")
        raise RuntimeError(f"Error loading song data: {e}")

    return model, passage_embeddings, song_data


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

                    # Store recommendations and similarities in session state
                    st.session_state.recommendations = retrieve_top_k_entries(
                        song_data, top_k_indices
                    )
                    st.session_state.similarities = top_k_similarities

                    toc = time.time()
                    st.success(f"Time taken: {toc - tic:.2f} seconds")

                except Exception as e:
                    st.error(f"Error computing similarities: {e}")

    # Display recommendations if they exist in session state
    if hasattr(st.session_state, "recommendations"):
        st.subheader("Your Recommendations:")

        # Create two columns: one for cards, one for lyrics display
        col1, col2 = st.columns([2, 3])

        # Initialize session state for selected lyrics if not exists
        if "selected_lyrics" not in st.session_state:
            st.session_state.selected_lyrics = None

        with col1:
            for i, song in enumerate(st.session_state.recommendations.itertuples(), 1):
                score = st.session_state.similarities[i - 1]
                # Create a card-like container for each song
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            padding: 1rem;
                            border-radius: 0.5rem;
                            background-color: #f0f2f6;
                            margin-bottom: 1rem;
                            cursor: pointer;
                            transition: transform 0.2s;
                            ">
                            <h4 style="margin: 0;">{i}. {song.song}</h4>
                            <p style="margin: 0.5rem 0;">artist: {song.artist}</p>
                            <p style="margin: 0; font-size: 0.8rem;">score: {score:.3f}</p>
                            <a href="{song.link}" target="_blank" style="font-size: 0.8rem;">More Info ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Button to show lyrics
                    if st.button("Show Lyrics", key=f"btn_{i}"):
                        st.session_state.selected_lyrics = {
                            "title": song.song,
                            "artist": song.artist,
                            "text": song.text,
                        }

        # Display lyrics in the second column
        with col2:
            st.markdown(
                """
                <div style="position: sticky; top: 0;">
                    <h3>Lyrics</h3>
                </div>
            """,
                unsafe_allow_html=True,
            )

            if st.session_state.selected_lyrics:
                st.markdown(f"**{st.session_state.selected_lyrics['title']}**")
                st.markdown(f"*by {st.session_state.selected_lyrics['artist']}*")
                st.markdown(st.session_state.selected_lyrics["text"])
            else:
                st.markdown("*Click 'Show Lyrics' on any song to view its lyrics here*")
