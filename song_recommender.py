import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# Model and data paths
MODEL_NAME = "jinaai/jina-embeddings-v3"  # Correct Hugging Face model identifier
EMBEDDINGS_PATH = './embeddings/jina-embeddings-v3_retrieval.passage.npy'
DATA_PATH = './data/spotify_millsongdata.csv'

@st.cache_resource
def load_model_and_data():
    """Load the model, tokenizer, embeddings, and dataset."""
    import os

    # Verify file existence
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {os.path.abspath(EMBEDDINGS_PATH)}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {os.path.abspath(DATA_PATH)}")

    # Load model and tokenizer
    try:
        st.info(f"Loading model: {MODEL_NAME}")
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        st.success("Model and tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        raise

    # Load embeddings
    try:
        passage_embeddings = np.load(EMBEDDINGS_PATH)
        st.success(f"Passage embeddings loaded: shape {passage_embeddings.shape}")
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

    return model, tokenizer, passage_embeddings, df

def compute_similarity(query_texts, model, tokenizer, passage_embeddings):
    """Compute cosine similarity between user input and song embeddings."""
    inputs = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    # Extract query embeddings
    query_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Debugging: Print shapes
    st.write(f"Query embeddings shape: {query_embeddings.shape}")
    st.write(f"Passage embeddings shape: {passage_embeddings.shape}")

    # Ensure proper dimensions for similarity computation
    if query_embeddings.ndim == 2 and query_embeddings.shape[0] == 1:
        query_embeddings = query_embeddings[0]  # Remove the batch dimension (shape becomes 1D)

    # Compute similarities
    similarities = passage_embeddings @ query_embeddings  # Transpose not needed for dot product
    return similarities

def get_top_k_indices(similarities, k=5):
    """Retrieve top-k indices for the most similar songs."""
    top_k_indices = np.argpartition(-similarities, k)[:k]
    sorted_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
    return sorted_indices

def retrieve_top_k_entries(df, indices):
    """Retrieve rows of songs based on indices."""
    return df.iloc[indices]

# Streamlit UI
st.title("🎵 Song Recommendation System")
st.markdown(
    """
    Welcome! Input a mood, description, or any text, and get personalized song recommendations!
    """
)

try:
    model, tokenizer, passage_embeddings, song_data = load_model_and_data()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# User Input
query = st.text_input("Enter a description of the song you're looking for:")
num_recommendations = st.slider("How many recommendations would you like?", 1, 10, 5)

if st.button("Recommend Songs"):
    if query.strip():
        with st.spinner("Finding songs for you..."):
            try:
                # Compute similarities
                similarities = compute_similarity([query], model, tokenizer, passage_embeddings)

                # Get top-k indices
                top_k_indices = get_top_k_indices(similarities, k=num_recommendations)

                # Retrieve recommendations
                recommendations = retrieve_top_k_entries(song_data, top_k_indices)

                st.subheader("Your Recommendations:")
                for i, song in enumerate(recommendations.itertuples(), 1):
                    st.markdown(
                        f"""
                        **{i}. {song.artist} - {song.song}**
                        - Lyrics Snippet: {song.text[:200]}...
                        - [More Info]({song.link})
                        """
                    )
            except Exception as e:
                st.error(f"Error during recommendation generation: {e}")
    else:
        st.warning("Please enter a query to get recommendations.")
