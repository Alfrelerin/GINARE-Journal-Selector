"""Carga el índice de revistas, artículos y embeddings desde data/,
cruzando todas las fuentes disponibles (JCR, SJR, DOAJ, Sherpa, review times)
y aplicando los overrides verificados por el usuario.

Diseño robusto: cada fuente puede faltar y la app sigue funcionando.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (
    ARTICLES_INDEX_PATH,
    DOAJ_INDEX_PATH,
    EMBEDDINGS_META_PATH,
    EMBEDDINGS_PATH,
    FAVORITES_PATH,
    JCR_INDEX_PATH,
    JCR_RANKS_PATH,
    JOURNALS_INDEX_PATH,
    REVIEW_TIMES_PATH,
    SHERPA_INDEX_PATH,
    SJR_INDEX_PATH,
)
from core.overrides import merge_overrides_into_journals


@dataclass
class IndexSources:
    """Qué fuentes están cargadas (para mostrar badges en la cabecera)."""
    journals: bool = False
    articles: bool = False
    embeddings: bool = False
    jcr: bool = False
    jcr_ranks: bool = False
    sjr: bool = False
    doaj: bool = False
    sherpa: bool = False
    review_times: bool = False
    overrides_count: int = 0


@dataclass
class JournalIndex:
    journals: pd.DataFrame                 # una fila por revista (consolidada)
    jcr_ranks: pd.DataFrame                # detalle por (revista, categoría)
    articles: pd.DataFrame                 # corpus de artículos
    embeddings: np.ndarray                 # matriz (n_artículos, dim)
    embeddings_meta: pd.DataFrame
    favorites: set[str]
    sources: IndexSources

    def n_journals(self) -> int:
        return len(self.journals)

    def n_articles(self) -> int:
        return len(self.articles)

    @property
    def has_jcr_data(self) -> bool:
        return self.sources.jcr

    def journal_row(self, journal_id: str) -> pd.Series | None:
        rows = self.journals[self.journals["journal_id"] == journal_id]
        return rows.iloc[0] if len(rows) else None

    def categories_for(self, issn: str | None) -> pd.DataFrame:
        """Devuelve las filas de jcr_ranks correspondientes a este ISSN
        (una por categoría)."""
        if not issn or self.jcr_ranks.empty:
            return pd.DataFrame()
        return self.jcr_ranks[self.jcr_ranks["issn"] == issn].copy()


# ── Utilidades ───────────────────────────────────────────────────────────

def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def _safe_read_npy(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 0), dtype=np.float32)
    return np.load(path)


def load_favorites() -> set[str]:
    if not FAVORITES_PATH.exists():
        return set()
    try:
        return set(json.loads(FAVORITES_PATH.read_text(encoding="utf-8")).get("favorites", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_favorites(favorites: set[str]) -> None:
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAVORITES_PATH.write_text(
        json.dumps({"favorites": sorted(favorites)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Carga principal ──────────────────────────────────────────────────────

def load_index() -> JournalIndex:
    sources = IndexSources()

    journals = _safe_read_parquet(JOURNALS_INDEX_PATH)
    sources.journals = not journals.empty
    articles = _safe_read_parquet(ARTICLES_INDEX_PATH)
    sources.articles = not articles.empty
    embeddings = _safe_read_npy(EMBEDDINGS_PATH)
    embeddings_meta = _safe_read_parquet(EMBEDDINGS_META_PATH)
    sources.embeddings = embeddings.size > 0 and not embeddings_meta.empty

    # ── Cruces con fuentes externas ──
    if not journals.empty:
        # JCR (privado, local)
        jcr = _safe_read_parquet(JCR_INDEX_PATH)
        sources.jcr = not jcr.empty
        if sources.jcr:
            journals = _merge_on_issn(journals, jcr, ["jcr_jif", "jcr_quartile", "jcr_category",
                                                      "jcr_best_rank", "jcr_best_percentile"])

        # SJR (abierto)
        sjr = _safe_read_parquet(SJR_INDEX_PATH)
        sources.sjr = not sjr.empty
        if sources.sjr:
            journals = _merge_on_issn(journals, sjr,
                                      ["sjr_score", "sjr_quartile", "sjr_h_index", "sjr_categories"])

        # DOAJ (abierto)
        doaj = _safe_read_parquet(DOAJ_INDEX_PATH)
        sources.doaj = not doaj.empty
        if sources.doaj:
            doaj_cols = [c for c in doaj.columns if c != "issn"]
            journals = _merge_on_issn(journals, doaj, doaj_cols)

        # Sherpa Romeo (requiere API key del usuario, opcional)
        sherpa = _safe_read_parquet(SHERPA_INDEX_PATH)
        sources.sherpa = not sherpa.empty and "sherpa_indexed" in sherpa.columns
        if sources.sherpa:
            sherpa_cols = [c for c in sherpa.columns if c != "issn"]
            journals = _merge_on_issn(journals, sherpa, sherpa_cols)

        # Tiempos de revisión (cómputo automático desde CrossRef)
        review_times = _safe_read_parquet(REVIEW_TIMES_PATH)
        sources.review_times = not review_times.empty
        if sources.review_times and "journal_id" in journals.columns:
            keep = [c for c in review_times.columns if c != "journal_id"]
            journals = journals.merge(review_times, on="journal_id", how="left", suffixes=("", "_rt"))

        # ── Overrides verificados manualmente ──
        journals = merge_overrides_into_journals(journals)
        # Cuenta cuántas revistas tienen verificación
        if "verified_at" in journals.columns:
            sources.overrides_count = int(journals["verified_at"].notna().sum())

    # JCR ranks (multi-categoría)
    jcr_ranks = _safe_read_parquet(JCR_RANKS_PATH)
    sources.jcr_ranks = not jcr_ranks.empty

    # Añade una columna `jcr_categories_list` a journals para que los filtros
    # puedan restringir por categoría sin tener que joinear de nuevo en cada
    # iteración.
    if sources.jcr_ranks and not journals.empty:
        cat_by_issn = (
            jcr_ranks.groupby("issn")["jcr_category"]
            .apply(lambda s: sorted({str(x) for x in s if isinstance(x, str)}))
            .to_dict()
        )

        def _gather_cats(row) -> list[str]:
            issns: set[str] = set()
            primary = row.get("issn")
            if primary and isinstance(primary, str):
                issns.add(primary)
            raw = row.get("all_issns", "") or ""
            for alt in str(raw).split(";"):
                alt = alt.strip()
                if alt:
                    issns.add(alt)
            collected: set[str] = set()
            for i in issns:
                collected.update(cat_by_issn.get(i, []))
            return sorted(collected)

        journals = journals.copy()
        journals["jcr_categories_list"] = journals.apply(_gather_cats, axis=1)

    return JournalIndex(
        journals=journals,
        jcr_ranks=jcr_ranks,
        articles=articles,
        embeddings=embeddings.astype(np.float32, copy=False) if embeddings.size else embeddings,
        embeddings_meta=embeddings_meta,
        favorites=load_favorites(),
        sources=sources,
    )


def _compute_jcr_categories_list(journals: pd.DataFrame, jcr_ranks: pd.DataFrame) -> pd.DataFrame:
    """Añade/actualiza la columna `jcr_categories_list` en journals a partir
    de la tabla jcr_ranks (una categoría por fila)."""
    if jcr_ranks is None or jcr_ranks.empty or journals.empty:
        return journals
    cat_by_issn = (
        jcr_ranks.groupby("issn")["jcr_category"]
        .apply(lambda s: sorted({str(x) for x in s if isinstance(x, str)}))
        .to_dict()
    )

    def _gather_cats(row) -> list[str]:
        issns: set[str] = set()
        primary = row.get("issn")
        if primary and isinstance(primary, str):
            issns.add(primary)
        raw = row.get("all_issns", "") or ""
        for alt in str(raw).split(";"):
            alt = alt.strip()
            if alt:
                issns.add(alt)
        collected: set[str] = set()
        for i in issns:
            collected.update(cat_by_issn.get(i, []))
        return sorted(collected)

    journals = journals.copy()
    journals["jcr_categories_list"] = journals.apply(_gather_cats, axis=1)
    return journals


def apply_uploaded_jcr(
    base: "JournalIndex",
    jcr_index_df: pd.DataFrame | None,
    jcr_ranks_df: pd.DataFrame | None,
) -> "JournalIndex":
    """Devuelve una copia del índice con un JCR subido por el usuario aplicado
    (cuartiles, IF y rankings). No toca el disco: es por sesión, así cada
    persona usa su propio JCR sin que viaje en el repo (licencia Clarivate)."""
    import copy as _copy

    journals = base.journals.copy()
    if jcr_index_df is not None and not jcr_index_df.empty:
        journals = _merge_on_issn(
            journals, jcr_index_df,
            ["jcr_jif", "jcr_quartile", "jcr_category", "jcr_best_rank", "jcr_best_percentile"],
        )

    jcr_ranks = (
        jcr_ranks_df if (jcr_ranks_df is not None and not jcr_ranks_df.empty)
        else base.jcr_ranks
    )
    journals = _compute_jcr_categories_list(journals, jcr_ranks)

    sources = _copy.copy(base.sources)
    sources.jcr = jcr_index_df is not None and not jcr_index_df.empty
    sources.jcr_ranks = jcr_ranks is not None and not jcr_ranks.empty

    return JournalIndex(
        journals=journals,
        jcr_ranks=jcr_ranks if jcr_ranks is not None else base.jcr_ranks,
        articles=base.articles,
        embeddings=base.embeddings,
        embeddings_meta=base.embeddings_meta,
        favorites=base.favorites,
        sources=sources,
    )


def _merge_on_issn(
    base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Merge robusto que intenta primero por ISSN primaria y luego por
    cualquier ISSN secundaria en `all_issns` (separadas por ;).

    Esto resuelve el caso típico: OpenAlex devuelve la eISSN pero JCR/SJR/DOAJ
    indexa por la print ISSN (o viceversa). Sin esto perderíamos ~80% de los
    matches.

    Si una columna ya existía en base, la sobrescribe con el valor de extra
    cuando extra tiene dato; si no, la añade.
    """
    if "issn" not in base.columns or "issn" not in extra.columns:
        return base

    cols_present = [c for c in columns if c in extra.columns]
    if not cols_present:
        return base

    extra_sub = extra[["issn"] + cols_present].drop_duplicates(subset=["issn"])
    extra_indexed = extra_sub.set_index("issn")
    extra_issns = set(extra_indexed.index.dropna())

    def find_matching_issn(row) -> str | None:
        primary = row.get("issn")
        if primary and primary in extra_issns:
            return primary
        # Probar todas las ISSN alternativas de all_issns
        raw = row.get("all_issns", "") or ""
        for alt in str(raw).split(";"):
            alt = alt.strip()
            if alt and alt in extra_issns:
                return alt
        return None

    matched = base.apply(find_matching_issn, axis=1)

    new_cols: dict[str, list] = {c: [] for c in cols_present}
    for issn_key in matched:
        if issn_key is not None and issn_key in extra_indexed.index:
            row_ext = extra_indexed.loc[issn_key]
            # Si hay duplicados, take first
            if isinstance(row_ext, pd.DataFrame):
                row_ext = row_ext.iloc[0]
            for c in cols_present:
                new_cols[c].append(row_ext.get(c))
        else:
            for c in cols_present:
                new_cols[c].append(None)

    result = base.copy()
    for c in cols_present:
        new_series = pd.Series(new_cols[c], index=result.index)
        if c in result.columns:
            # Combine: nuevos valores ganan cuando son no-nulos; en otro caso, mantén lo existente
            result[c] = new_series.combine_first(result[c])
        else:
            result[c] = new_series
    return result
