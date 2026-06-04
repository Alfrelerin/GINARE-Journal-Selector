"""Calcula los embeddings SPECTER de todos los artículos del corpus y los
guarda en data/embeddings.npy + data/embeddings_meta.parquet.

**INCREMENTAL**: solo embebe artículos que aún no estén en embeddings_meta.
Esto permite ampliar el universo de revistas (con `expand_to_jcr.py`) sin
volver a embeber lo que ya tenías.

Tiempos típicos:
  - CPU: 3-5 art/s   → 100k artículos ≈ 5-9 horas
  - MPS (Apple M):   ~10-15 art/s → 100k artículos ≈ 2-3 horas
  - GPU (CUDA):      ~50-100 art/s → 100k artículos ≈ <40 min

El proceso es resumible: si lo cortas en mitad, al volverlo a lanzar
continúa donde lo dejaste (porque ya habrá guardado checkpoints).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import (
    ARTICLES_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_META_PATH,
    EMBEDDINGS_PATH,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHECKPOINT_EVERY_BATCHES = 50   # guarda parcial cada 50 batches (≈ 800 art)


def main(batch_size: int = 16) -> None:
    if not ARTICLES_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No existe {ARTICLES_INDEX_PATH}. Ejecuta antes: "
            "python -m pipelines.fetch_openalex"
        )

    articles = pd.read_parquet(ARTICLES_INDEX_PATH)
    log.info("Corpus total: %d artículos", len(articles))

    # ── Carga incremental ──
    if EMBEDDINGS_META_PATH.exists() and EMBEDDINGS_PATH.exists():
        existing_meta = pd.read_parquet(EMBEDDINGS_META_PATH)
        existing_emb = np.load(EMBEDDINGS_PATH)
        already_embedded = set(existing_meta["article_id"].astype(str))
        log.info("Ya embebidos: %d artículos. Solo se procesará lo nuevo.",
                 len(already_embedded))
    else:
        existing_meta = pd.DataFrame(columns=["article_id", "journal_id"])
        existing_emb = np.zeros((0, 0), dtype=np.float32)
        already_embedded = set()

    new_articles = articles[~articles["article_id"].astype(str).isin(already_embedded)].copy()
    n_new = len(new_articles)
    log.info("Artículos por embeber: %d", n_new)
    if n_new == 0:
        log.info("✔ Nada nuevo que embeber. Sales sin cambios.")
        return

    # ── Detección de dispositivo ──
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log.info("Dispositivo: %s", device)

    log.info("Cargando modelo %s …", EMBEDDING_MODEL_NAME)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    texts = (new_articles["title"].fillna("") + ". " + new_articles["abstract"].fillna("")).tolist()

    new_vecs: list[np.ndarray] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    log.info("Procesando %d batches de %d…", n_batches, batch_size)

    def _checkpoint():
        if not new_vecs:
            return
        new_matrix = np.vstack(new_vecs)
        # Concatena con el existente
        if existing_emb.size > 0 and existing_emb.shape[1] == new_matrix.shape[1]:
            final_emb = np.vstack([existing_emb, new_matrix])
        else:
            final_emb = new_matrix
        # Mantén meta alineada
        embedded_so_far = new_articles.iloc[: len(new_matrix)][["article_id", "journal_id"]]
        final_meta = pd.concat([existing_meta, embedded_so_far], ignore_index=True)

        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, final_emb)
        final_meta.to_parquet(EMBEDDINGS_META_PATH, index=False)
        log.info("💾 Checkpoint: %d vectores totales (%d nuevos esta ejecución)",
                 len(final_emb), len(new_matrix))

    try:
        for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size))):
            batch = texts[i : i + batch_size]
            emb = model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False,
                normalize_embeddings=True,
            ).astype(np.float32)
            new_vecs.append(emb)

            if (batch_idx + 1) % CHECKPOINT_EVERY_BATCHES == 0:
                _checkpoint()
    finally:
        _checkpoint()

    log.info("✔ Embeddings completos.")


if __name__ == "__main__":
    main()
