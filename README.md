# YouTube RAG Chatbot

## Run locally
```bash
pip install -r requirements.txt
mkdir -p .streamlit
echo 'HUGGINGFACEHUB_API_TOKEN = "hf_your_new_token"' > .streamlit/secrets.toml
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Go to share.streamlit.io → New app → pick the repo/branch → main file `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   HUGGINGFACEHUB_API_TOKEN = "hf_your_new_token"
   ```
4. Deploy. First video load will be slower (embedding model download); subsequent ones are cached.

⚠️ Regenerate your Hugging Face token before deploying — the one in the
original notebook was exposed and should be revoked at
https://huggingface.co/settings/tokens.

## Further improvements worth considering
- **Hybrid retrieval (BM25 + FAISS)** — catches exact keyword matches (names, numbers) that pure semantic search sometimes misses.
- **Cross-encoder reranking** — re-score the top-k retrieved chunks for relevance before passing to the LLM; noticeably improves answer precision.
- **MMR retrieval** instead of plain similarity, to reduce redundant/overlapping chunks in context.
- **Multi-video support** — let a user load several videos into one session and query across them, or compare videos.
- **Non-English / auto-translated captions** — currently falls back to "any language," but doesn't translate; could pipe through a translation step.
- **Evaluation harness (Ragas)** — track faithfulness/answer-relevance as you tune chunk size, k, or prompts, instead of eyeballing responses.
- **Persist vector stores to disk** (e.g., keyed by video ID) so re-loading a previously seen video is instant instead of re-embedding.
