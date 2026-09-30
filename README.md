# Emoji Semantic Search (THA-1, Question 1)

Type a feeling or situation (e.g. *"celebrating a big win"*) and get the 5 best-matching emojis with similarity scores.
Embeddings come from the pretrained `all-MiniLM-L6-v2` model; cosine similarity, ranking and Precision@5 are implemented by hand with NumPy.

| File | What it is |
|---|---|
| `emoji_semantic_search.ipynb` | The submission notebook: data → embeddings → similarity → retrieval → evaluation → assessment → discussion → resources |
| `emojis.csv` | Dataset: 1,458 emojis built from Unicode CLDR annotations (columns `emoji, name, keywords, text`) |

## How to run

### Option A: Google Colab (easiest)
1. Upload `emoji_semantic_search.ipynb` **and** `emojis.csv` to Colab (Files panel on the left).
2. *Runtime → Run all*. The first cell installs `sentence-transformers`; the model (~90 MB) downloads automatically.

If `emojis.csv` is missing, the notebook rebuilds it from the Unicode sources (set `REBUILD = True` to force this).
The Unicode files can change over time, so a rebuild may give slightly different counts than the saved CSV.

### Option B: locally (Windows)
```bash
python -m venv C:\Users\<you>\.venvs\tha1
C:\Users\<you>\.venvs\tha1\Scripts\python.exe -m pip install sentence-transformers pandas notebook
C:\Users\<you>\.venvs\tha1\Scripts\python.exe -m notebook emoji_semantic_search.ipynb
```
Then *Kernel → Restart & Run All*.

**Two Windows problems we hit, and their fixes:**
- **`OSError: [WinError 206] The filename or extension is too long`** during `pip install`: the Microsoft Store version of Python lives in a very long folder path, and PyTorch's files push it past Windows' 260-character limit. Fix: create the virtual environment in a **short** path as shown above (or enable Windows long-path support).
- **`CERTIFICATE_VERIFY_FAILED` when downloading the model**: antivirus software that scans HTTPS traffic (here: Avast) re-signs it with its own certificate, which Python's default certificate list does not trust. Fix, inside the virtual environment only:
  ```bash
  C:\Users\<you>\.venvs\tha1\Scripts\python.exe -m pip install truststore
  ```
  and create `C:\Users\<you>\.venvs\tha1\Lib\site-packages\sitecustomize.py` containing:
  ```python
  import truststore
  truststore.inject_into_ssl()
  ```
  This makes Python use the Windows certificate store. Colab does not need this.

## Notes
- Runtime: about 1 minute on a laptop CPU (embedding 1,458 short texts takes ~20 s).
- Result: average Precision@5 of **0.64** over 10 test queries (clear emotions, situations, a mixed emotion, slang, negation and an abstract query), judged manually.
