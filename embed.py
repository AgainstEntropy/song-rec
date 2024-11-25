import numpy as np
import pandas as pd
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm
from transformers import AutoModel

from xlm_roberta.modeling_lora import XLMRobertaLoRA

if __name__ == '__main__':

    model_name = 'jina-embeddings-v3'
    model_folder = f'./models/{model_name}/'

    # Initialize the model
    print("Loading model...")
    model = AutoModel.from_pretrained(model_folder, trust_remote_code=True)
    model: XLMRobertaLoRA

    if is_cuda_available():
        model.to('cuda')

    print("Loading data...")
    data_path = './data/spotify_millsongdata.csv'
    df = pd.read_csv(data_path)
    lyrics = df["text"]

    print("Encoding...")
    batch_size = 64
    embeddings_list = []
    task = "retrieval.passage"

    for start in tqdm(range(0, len(lyrics), batch_size), desc="Encoding chunks"):
        end = start + batch_size
        batch = lyrics.iloc[start:end].to_list()
        batch_embeddings = model.encode(batch, task=task, batch_size=batch_size)
        embeddings_list.append(batch_embeddings)

    # Merge all chunk embeddings into a single ndarray
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    print(all_embeddings.shape)

    print("Saving...")
    # Save embeddings to a file
    save_path = f'./embeddings/{model_name}_{task}.npy'
    np.save(save_path, all_embeddings)
    print(f'Saved embeddings to {save_path}')
