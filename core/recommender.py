"""Motor de recomendación: embebe título+abstract del usuario y puntúa cada
revista por similitud al corpus de sus artículos recientes.

Estrategia de scoring:
  - similitud semántica = media de los top-k coseno entre tu artículo y los
    artículos más parecidos publicados en la revista (top-k, no media global,
    para no penalizar revistas amplias).
  - score final = combinación ponderada de:
        similitud semántica  · w_topic        (por defecto 0.65)
        cuartil normalizado  · w_quartile     (por defecto 0.15)
        IF normalizado       · w_if           (por defecto 0.10)
        tiempo de revisión   · w_speed        (por defecto 0.05)
        coste invertido      · w_cost         (por defecto 0.05)
  - los pesos son configurables desde la UI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from core.config import (
    DEFAULT_MIN_ARTICLES,
    DEFAULT_RANKING_RESULTS,
    DEFAULT_TOP_K_ARTICLES,
    EMBEDDING_MODEL_NAME,
)
from core.data_loader import JournalIndex
from core.filters import QUARTILE_ORDER

log = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    topic: float = 0.65
    quartile: float = 0.15
    impact: float = 0.10
    speed: float = 0.05
    cost: float = 0.05

    def normalised(self) -> "ScoringWeights":
        total = self.topic + self.quartile + self.impact + self.speed + self.cost
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            topic=self.topic / total,
            quartile=self.quartile / total,
            impact=self.impact / total,
            speed=self.speed / total,
            cost=self.cost / total,
        )


@dataclass
class RecommendationRequest:
    title: str
    abstract: str
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    top_k_articles: int = DEFAULT_TOP_K_ARTICLES
    min_articles_per_journal: int = DEFAULT_MIN_ARTICLES
    max_results: int = DEFAULT_RANKING_RESULTS


@dataclass
class JournalRecommendation:
    journal_id: str
    name: str
    issn: str | None
    publisher: str | None
    homepage_url: str | None
    quartile: str | None
    impact_factor: float | None
    apc_eur: float | None
    oa_model: str | None             # diamond | gold | hybrid | subscription | unknown
    time_to_first_decision_weeks: float | None
    time_to_publication_weeks: float | None
    acceptance_rate_pct: float | None
    topic_similarity: float          # 0..1
    quartile_score: float            # 0..1
    impact_score: float              # 0..1
    speed_score: float               # 0..1
    cost_score: float                # 0..1
    final_score: float               # 0..1
    is_favorite: bool
    verified_fields: list[str]       # qué campos están verificados por el usuario
    verified_at: str | None
    jcr_categories: list[dict]       # lista de {category, quartile, rank, total}
    top_articles: list[dict]


# ── Modelo de embeddings ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """Carga perezosa de SPECTER. Solo importa torch/sentence-transformers si
    se llama, para que la app pueda arrancar aunque las dependencias pesadas
    aún se estén instalando."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ¿Está el modelo ya descargado en la caché local de HuggingFace?
    #  · En local (tu Mac) SÍ → forzamos modo offline para evitar que la carga
    #    se cuelgue esperando una comprobación de red a HuggingFace.
    #  · En la nube / primera vez NO → dejamos que lo descargue (online).
    hf_home = os.environ.get("HF_HOME")
    hf_cache = (
        os.environ.get("HF_HUB_CACHE")
        or (os.path.join(hf_home, "hub") if hf_home else None)
        or os.path.expanduser("~/.cache/huggingface/hub")
    )
    cached = os.path.isdir(os.path.join(hf_cache, "models--allenai--specter"))
    mode = "1" if cached else "0"
    os.environ.setdefault("HF_HUB_OFFLINE", mode)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", mode)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers no está instalado. Ejecuta: "
            "uv pip install sentence-transformers"
        ) from exc

    log.info("Cargando modelo de embeddings %s (%s)…",
             EMBEDDING_MODEL_NAME, "offline/caché" if cached else "descarga")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    log.info("Modelo %s cargado.", EMBEDDING_MODEL_NAME)
    return model


def embed_query(title: str, abstract: str) -> np.ndarray:
    """Embedding SPECTER del título+abstract del usuario.

    SPECTER se entrena con el formato '[TITLE] [SEP] [ABSTRACT]' (la lib
    sentence-transformers gestiona el separador internamente al pasar la
    cadena ya unida)."""
    text = (title.strip() + ". " + abstract.strip()).strip()
    if not text:
        raise ValueError("Título y abstract vacíos.")
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


# ── Ranking ───────────────────────────────────────────────────────────────

def _cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k coseno asumiendo query ya normalizado. Devuelve (sims, indices)."""
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    # Si la matriz no está normalizada, la normalizamos en caliente
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_n = matrix / norms
    sims = matrix_n @ query
    k_eff = min(k, sims.shape[0])
    if k_eff <= 0:
        return np.array([]), np.array([], dtype=int)
    idx_part = np.argpartition(-sims, k_eff - 1)[:k_eff]
    idx_sorted = idx_part[np.argsort(-sims[idx_part])]
    return sims[idx_sorted], idx_sorted


def _normalise_quartile_score(q: str | None) -> float:
    if q is None or pd.isna(q):
        return 0.5  # neutro si no se conoce
    rank = QUARTILE_ORDER.get(str(q), 4)
    return (5 - rank) / 4.0  # Q1→1.0, Q2→0.75, Q3→0.5, Q4→0.25


def _normalise_impact_score(if_value: float | None, p95: float) -> float:
    if if_value is None or pd.isna(if_value) or if_value <= 0:
        return 0.0
    # log-escalado y comprimido al percentil 95 para que un IF=20 no aplaste
    return float(min(1.0, np.log1p(if_value) / np.log1p(max(p95, 1.0))))


def _normalise_speed_score(weeks: float | None) -> float:
    if weeks is None or pd.isna(weeks):
        return 0.5
    # Para tiempo a primera decisión: 4 semanas → 1.0; 26+ semanas → 0.0
    return float(max(0.0, min(1.0, (26 - weeks) / 22.0)))


def _normalise_speed_publication(weeks: float | None) -> float:
    """Variante para tiempo a publicación (suele ser mayor): 8s → 1.0, 52s → 0.0."""
    if weeks is None or pd.isna(weeks):
        return 0.5
    return float(max(0.0, min(1.0, (52 - weeks) / 44.0)))


def _normalise_cost_score(apc_eur: float | None) -> float:
    if apc_eur is None or pd.isna(apc_eur) or apc_eur == 0:
        return 1.0  # gratis o no especificado: mejor
    # 0€ → 1.0; 4000€ → 0.0
    return float(max(0.0, min(1.0, 1.0 - (apc_eur / 4000.0))))


def rank_journals(
    request: RecommendationRequest,
    index: JournalIndex,
    candidate_journal_ids: list[str] | None = None,
) -> list[JournalRecommendation]:
    """Calcula el ranking. `candidate_journal_ids` permite restringir el
    universo (típicamente después de aplicar filtros)."""

    if index.embeddings.size == 0 or index.embeddings_meta.empty:
        log.warning("El índice está vacío. Corre `python pipelines/run_all.py` primero.")
        return []

    weights = request.weights.normalised()
    query_vec = embed_query(request.title, request.abstract)

    # Restringir corpus a las revistas candidatas (si se pasan)
    meta = index.embeddings_meta
    if candidate_journal_ids is not None:
        cand_set = set(candidate_journal_ids)
        mask = meta["journal_id"].isin(cand_set).to_numpy()
        if not mask.any():
            return []
        meta = meta.loc[mask].reset_index(drop=True)
        sub_embeddings = index.embeddings[mask]
    else:
        sub_embeddings = index.embeddings

    # Calculamos similitud de TODOS los artículos al query (es matrix-vector,
    # rápido incluso con 100k artículos)
    norms = np.linalg.norm(sub_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims_all = (sub_embeddings / norms) @ query_vec  # vector (n_articles,)

    meta = meta.assign(_sim=sims_all)

    # Agregamos por revista: media de los top-k más parecidos
    def _agg(group: pd.DataFrame) -> pd.Series:
        if len(group) < request.min_articles_per_journal:
            return pd.Series({"topic_sim": np.nan, "n": len(group)})
        topk = group.nlargest(request.top_k_articles, "_sim")
        return pd.Series({"topic_sim": float(topk["_sim"].mean()), "n": len(group)})

    agg = meta.groupby("journal_id").apply(_agg, include_groups=False).reset_index()
    agg = agg.dropna(subset=["topic_sim"])

    # Unimos con la tabla de revistas
    journals_df = index.journals
    merged = agg.merge(journals_df, on="journal_id", how="inner")
    if merged.empty:
        return []

    # Para normalizar el IF necesitamos el p95
    if_series = merged.get("jcr_jif")
    if if_series is None or if_series.isna().all():
        if_series = merged.get("openalex_2yr_mean_citedness", pd.Series(dtype=float))
    p95 = float(np.nanpercentile(if_series.dropna(), 95)) if len(if_series.dropna()) else 10.0

    recommendations: list[JournalRecommendation] = []
    for _, row in merged.iterrows():
        # Cuartil efectivo: override manual > JCR oficial.
        q_override = row.get("quartile")
        if q_override is not None and not pd.isna(q_override) and str(q_override).startswith("Q"):
            q = str(q_override)
        else:
            jcr_q = row.get("jcr_quartile")
            q = jcr_q if jcr_q and not pd.isna(jcr_q) else None

        # IF efectivo: override manual > JCR JIF > proxy OpenAlex.
        if_val = row.get("impact_factor")
        if if_val is None or pd.isna(if_val):
            if_val = row.get("jcr_jif")
        if if_val is None or pd.isna(if_val):
            if_val = row.get("openalex_2yr_mean_citedness")

        # APC: prioridad override > DOAJ > OpenAlex
        apc = row.get("apc_eur")
        if pd.isna(apc) or apc is None:
            apc = row.get("doaj_apc_eur")

        # Tiempo a primera decisión (verificado) o a publicación (CrossRef)
        t_first = row.get("time_to_first_decision_weeks")
        t_pub = row.get("time_to_publication_weeks_median")

        # Para el speed_score usamos el dato verificado si existe;
        # si no, una versión "compacta" del tiempo a publicación.
        if t_first is not None and not pd.isna(t_first):
            speed_score = _normalise_speed_score(float(t_first))
        else:
            speed_score = _normalise_speed_publication(
                float(t_pub) if t_pub is not None and not pd.isna(t_pub) else None
            )

        pub = row.get("publisher")
        # Los campos vacíos llegan de pandas como NaN (float), no como None.
        # Normalizamos a str|None para que la UI y el export no fallen.
        pub = str(pub).strip() if pub is not None and not pd.isna(pub) else None
        oa_model = row.get("oa_model")
        oa_model = str(oa_model).strip() if oa_model is not None and not pd.isna(oa_model) else None
        topic_sim = float(row["topic_sim"])

        topic_score = (topic_sim + 1.0) / 2.0
        q_score = _normalise_quartile_score(q)
        imp_score = _normalise_impact_score(
            float(if_val) if if_val is not None and not pd.isna(if_val) else None,
            p95,
        )
        cost_score = _normalise_cost_score(apc if apc is not None and not pd.isna(apc) else None)

        final = (
            weights.topic * topic_score
            + weights.quartile * q_score
            + weights.impact * imp_score
            + weights.speed * speed_score
            + weights.cost * cost_score
        )

        # Top artículos similares
        jmeta = meta[meta["journal_id"] == row["journal_id"]]
        top_articles_df = jmeta.nlargest(request.top_k_articles, "_sim")
        articles_df = index.articles
        top_articles = []
        for _, am in top_articles_df.iterrows():
            art = articles_df[articles_df["article_id"] == am["article_id"]]
            if len(art):
                a = art.iloc[0]
                top_articles.append({
                    "title": a.get("title", ""),
                    "year": int(a["year"]) if not pd.isna(a.get("year")) else None,
                    "doi": a.get("doi"),
                    "similarity": float(am["_sim"]),
                })

        # Categorías JCR con posición.
        # Si el usuario fijó manualmente una categoría, esa reemplaza a las
        # automáticas (override). Si no, usamos las de jcr_ranks.
        issn_val = row.get("issn")
        jcr_cats: list[dict] = []
        cat_override = row.get("manual_jcr_category")
        if cat_override is not None and not pd.isna(cat_override) and str(cat_override).strip():
            rank_ov = row.get("manual_jcr_rank")
            total_ov = row.get("manual_jcr_total")
            jcr_cats.append({
                "category": str(cat_override).strip(),
                "quartile": q,
                "rank": int(rank_ov) if rank_ov is not None and not pd.isna(rank_ov) else None,
                "total": int(total_ov) if total_ov is not None and not pd.isna(total_ov) else None,
                "percentile": None,
            })
        else:
            cats_df = index.categories_for(issn_val) if issn_val else pd.DataFrame()
            if not cats_df.empty:
                for _, c in cats_df.iterrows():
                    jcr_cats.append({
                        "category": c.get("jcr_category"),
                        "quartile": c.get("jcr_quartile"),
                        "rank": int(c["jcr_rank"]) if pd.notna(c.get("jcr_rank")) else None,
                        "total": int(c["jcr_category_size"]) if pd.notna(c.get("jcr_category_size")) else None,
                        "percentile": float(c["jcr_percentile"]) if pd.notna(c.get("jcr_percentile")) else None,
                    })

        # Campos verificados
        verified_fields: list[str] = []
        for f in ("oa_model", "apc_eur", "time_to_first_decision_weeks",
                  "acceptance_rate_pct", "homepage_url",
                  "publisher", "impact_factor", "quartile", "manual_jcr_category"):
            if row.get(f"verified_{f}") is True:
                verified_fields.append(f)

        recommendations.append(JournalRecommendation(
            journal_id=row["journal_id"],
            name=row.get("name", ""),
            issn=issn_val,
            publisher=pub,
            homepage_url=row.get("homepage_url"),
            quartile=q,
            impact_factor=float(if_val) if if_val is not None and not pd.isna(if_val) else None,
            apc_eur=float(apc) if apc is not None and not pd.isna(apc) else None,
            oa_model=str(oa_model) if oa_model and not pd.isna(oa_model) else None,
            time_to_first_decision_weeks=float(t_first) if t_first is not None and not pd.isna(t_first) else None,
            time_to_publication_weeks=float(t_pub) if t_pub is not None and not pd.isna(t_pub) else None,
            acceptance_rate_pct=float(row.get("acceptance_rate_pct")) if row.get("acceptance_rate_pct") is not None and not pd.isna(row.get("acceptance_rate_pct")) else None,
            topic_similarity=topic_score,
            quartile_score=q_score,
            impact_score=imp_score,
            speed_score=speed_score,
            cost_score=cost_score,
            final_score=final,
            is_favorite=issn_val in index.favorites if issn_val else False,
            verified_fields=verified_fields,
            verified_at=row.get("verified_at"),
            jcr_categories=jcr_cats,
            top_articles=top_articles,
        ))

    recommendations.sort(key=lambda r: r.final_score, reverse=True)
    return recommendations[: request.max_results]
