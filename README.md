# Song-Rec

Clone this repo:

```bash
git clone --recurse-submodules https://github.com/ethan-z/song-rec.git
```

## Installation

```bash
pip install -r requirements.txt
```

To use Flash-Attention:

```bash
pip install flash-attn --no-build-isolation
```

## Download the model

```bash
huggingface-cli download jinaai/jina-embeddings-v3 --local-dir ./models/jina-embeddings-v3
```

## Embed the data

```bash
python embed.py
```

## Test query

Follow the blocks in [`test_query.ipynb`](./test_query.ipynb).
