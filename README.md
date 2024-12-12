# Song-Rec

Clone this repo:

```bash
git clone --recurse-submodules https://github.com/AgainstEntropy/song-rec.git
```

If for some reason the submodules are not cloned, run:

```bash
cd song-rec
git submodule update --init --recursive
```

## Installation

```bash
pip install -r requirements.txt
```

To use Flash-Attention:

```bash
pip install flash-attn --no-build-isolation
```

### Download the model

```bash
huggingface-cli download jinaai/jina-embeddings-v3 --local-dir ./models/jina-embeddings-v3
```

### Embed the data

You can either create the embeddings from the raw data:

```bash
python embed.py
```

Or download the pre-computed embeddings from [Google Drive](https://drive.google.com/drive/folders/1zcZejtGsIWJz39vXnr-XExmu6DWRItte?usp=sharing) and put it in `./embeddings/` folder.

## Test query

Follow the blocks in [`test_query.ipynb`](./test_query.ipynb).

## Run the app

```bash
streamlit run song_recommender.py
```
