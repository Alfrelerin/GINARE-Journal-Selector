"""Expande el universo de revistas candidatas a TODAS las que están en
tu JCR (data/jcr_index.parquet), no solo a las 59 semilla.

Es incremental y resumible: si lo interrumpes (Ctrl+C, cierre del terminal)
y lo vuelves a lanzar, solo procesa lo que falta. Las llamadas a OpenAlex
están cacheadas en disco.

Para cada ISSN del JCR:
  1) Si ya está cubierto por nuestro índice (issn o all_issns), se salta.
  2) Si no, resuelve en OpenAlex por ISSN.
  3) Deduplica por source_id de OpenAlex (porque print + eISSN apuntan al
     mismo source).
  4) Descarga ~N artículos recientes con abstract.
  5) Añade revista y artículos al índice.

Después corre:
    python -m pipelines.compute_embeddings
para embeber los artículos nuevos (también incremental).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from core.config import (
    ARTICLES_INDEX_PATH,
    JCR_INDEX_PATH,
    JOURNALS_INDEX_PATH,
    TARGET_SJR_CATEGORIES,
)
from pipelines.fetch_openalex import (
    _abstract_from_inverted,
    fetch_works,
    resolve_source,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_ARTICLES_PER_JOURNAL = 30   # menor que la semilla porque son muchas más
SAVE_EVERY_N = 25                   # guarda parciales cada N revistas procesadas


def _collect_existing_issns(journals: pd.DataFrame) -> set[str]:
    """Todas las ISSNs ya cubiertas por el índice (primaria + all_issns)."""
    existing = set(journals["issn"].dropna().astype(str))
    if "all_issns" in journals.columns:
        for s in journals["all_issns"].dropna():
            for i in str(s).split(";"):
                i = i.strip()
                if i:
                    existing.add(i)
    return existing


def _journal_from_source(src: dict, jcr_row: pd.Series, note: str = "") -> dict:
    oa_id = src.get("id", "")
    issns = src.get("issn") or []
    primary = src.get("issn_l") or (issns[0] if issns else jcr_row["issn"])
    summary = src.get("summary_stats", {}) or {}

    # APC: si está disponible en OpenAlex
    apc_eur = None
    apc_field = src.get("apc_usd") or src.get("apc_prices")
    if isinstance(apc_field, list) and apc_field:
        for entry in apc_field:
            if isinstance(entry, dict) and entry.get("price"):
                rate = 0.92 if entry.get("currency") == "USD" else 1.0
                apc_eur = float(entry["price"]) * rate
                break
    elif isinstance(apc_field, (int, float)):
        apc_eur = float(apc_field) * 0.92

    return {
        "journal_id": oa_id.split("/")[-1] if oa_id else jcr_row["name"],
        "name": src.get("display_name", jcr_row["name"]),
        "issn": primary,
        "all_issns": ";".join(issns) if issns else "",
        "publisher": src.get("host_organization_name"),
        "openalex_2yr_mean_citedness": summary.get("2yr_mean_citedness"),
        "openalex_h_index": summary.get("h_index"),
        "works_count": src.get("works_count"),
        "is_oa": src.get("is_oa"),
        "apc_eur": apc_eur,
        "review_weeks": None,
        "in_jcr": True,
        "in_target_category": True,
        "is_favorite": False,
        "homepage_url": src.get("homepage_url"),
        "note": note,
    }


def _articles_from_works(works: list[dict], journal_id: str) -> list[dict]:
    rows = []
    for w in works:
        abstract = _abstract_from_inverted(w.get("abstract_inverted_index"))
        if not (w.get("title") and abstract):
            continue
        rows.append({
            "article_id": w["id"].split("/")[-1],
            "journal_id": journal_id,
            "title": w.get("title", ""),
            "abstract": abstract,
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
        })
    return rows


def main(
    articles_per_journal: int = DEFAULT_ARTICLES_PER_JOURNAL,
    max_to_process: int | None = None,
    sleep_between: float = 0.05,
) -> None:
    if not JOURNALS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Falta {JOURNALS_INDEX_PATH}. Ejecuta antes: python -m pipelines.fetch_openalex"
        )
    if not JCR_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Falta {JCR_INDEX_PATH}. Coloca tus CSV en data/jcr/ y corre: "
            "python -m pipelines.build_jcr_list"
        )

    journals = pd.read_parquet(JOURNALS_INDEX_PATH)
    articles = pd.read_parquet(ARTICLES_INDEX_PATH) if ARTICLES_INDEX_PATH.exists() else pd.DataFrame()
    jcr = pd.read_parquet(JCR_INDEX_PATH)

    existing_issns = _collect_existing_issns(journals)
    existing_journal_ids = set(journals["journal_id"].astype(str))
    existing_article_ids = set(articles["article_id"].astype(str)) if "article_id" in articles.columns else set()

    log.info("Estado actual: %d revistas, %d artículos en el índice",
             len(journals), len(articles))
    log.info("ISSNs en JCR a evaluar: %d", len(jcr))

    # Ordenamos por JIF descendente para procesar primero las más relevantes;
    # así si interrumpes, lo que se ha embebido ya es lo de mayor IF.
    jcr_sorted = jcr.sort_values("jcr_jif", ascending=False, na_position="last").reset_index(drop=True)

    new_journals: list[dict] = []
    new_articles: list[dict] = []
    seen_source_ids: set[str] = set()
    n_skipped_existing = 0
    n_resolved = 0
    n_no_resolve = 0
    n_processed = 0

    def _checkpoint():
        if not new_journals and not new_articles:
            return
        nonlocal journals, articles
        journals = pd.concat([journals, pd.DataFrame(new_journals)], ignore_index=True) if new_journals else journals
        articles = pd.concat([articles, pd.DataFrame(new_articles)], ignore_index=True) if new_articles else articles
        # Quita duplicados conservando el primero
        journals.drop_duplicates(subset=["journal_id"], keep="first", inplace=True)
        articles.drop_duplicates(subset=["article_id"], keep="first", inplace=True)
        journals.to_parquet(JOURNALS_INDEX_PATH, index=False)
        articles.to_parquet(ARTICLES_INDEX_PATH, index=False)
        log.info("💾 Checkpoint guardado: %d revistas, %d artículos",
                 len(journals), len(articles))
        new_journals.clear()
        new_articles.clear()

    try:
        for idx, jrow in jcr_sorted.iterrows():
            if max_to_process and n_processed >= max_to_process:
                log.info("Tope max_to_process=%d alcanzado", max_to_process)
                break

            issn = jrow.get("issn")
            if not issn or pd.isna(issn):
                continue
            issn = str(issn)

            # Skip si ya la tenemos
            if issn in existing_issns:
                n_skipped_existing += 1
                continue

            name = jrow.get("name", "")
            try:
                src = resolve_source(name, issn)
            except Exception as exc:
                log.warning("  Fallo resolviendo %s (%s): %s", name, issn, exc)
                src = None

            if not src:
                n_no_resolve += 1
                continue

            oa_id = src.get("id", "")
            sid = oa_id.split("/")[-1] if oa_id else None
            if not sid:
                continue
            if sid in seen_source_ids or sid in existing_journal_ids:
                continue
            seen_source_ids.add(sid)
            n_resolved += 1
            n_processed += 1

            # Crear fila de revista
            jrow_dict = _journal_from_source(src, jrow, note=f"JCR · {jrow.get('jcr_category','')}")
            new_journals.append(jrow_dict)
            # Actualiza existing para deduplicar al vuelo
            existing_issns.add(jrow_dict["issn"])
            for ai in str(jrow_dict["all_issns"]).split(";"):
                if ai.strip():
                    existing_issns.add(ai.strip())

            # Artículos
            try:
                works = fetch_works(oa_id, articles_per_journal)
            except Exception as exc:
                log.warning("  Fallo trayendo works de %s: %s", name, exc)
                works = []

            arts = _articles_from_works(works, jrow_dict["journal_id"])
            for a in arts:
                if a["article_id"] not in existing_article_ids:
                    new_articles.append(a)
                    existing_article_ids.add(a["article_id"])

            if (idx + 1) % 10 == 0:
                log.info("[%5d/%d] %-40s → %d artículos (acum: %d revistas, %d artículos nuevos)",
                         idx + 1, len(jcr_sorted), name[:40], len(arts),
                         len(new_journals) + (len(journals) - n_skipped_existing), len(new_articles))

            # Checkpoint periódico
            if n_processed > 0 and n_processed % SAVE_EVERY_N == 0:
                _checkpoint()

            if sleep_between:
                time.sleep(sleep_between)
    finally:
        _checkpoint()

    log.info("=" * 60)
    log.info("Resumen: %d ya estaban, %d resueltas y añadidas, %d no resueltas en OpenAlex",
             n_skipped_existing, n_resolved, n_no_resolve)
    log.info("Índice final: %d revistas, %d artículos", len(journals), len(articles))
    log.info("Siguiente paso: python -m pipelines.compute_embeddings")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--articles", type=int, default=DEFAULT_ARTICLES_PER_JOURNAL,
                   help="Cuántos artículos por revista descargar")
    p.add_argument("--max", type=int, default=None,
                   help="Procesar solo las N primeras revistas no cubiertas (para test)")
    args = p.parse_args()
    main(articles_per_journal=args.articles, max_to_process=args.max)
