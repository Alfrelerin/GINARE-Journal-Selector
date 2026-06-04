"""Filtros aplicables al índice de revistas antes de rankear."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

QUARTILE_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# Modelos de OA que el usuario puede aceptar/excluir
OA_MODELS_ALL = ("diamond", "gold", "hybrid", "subscription", "unknown")


@dataclass
class JournalFilters:
    """Estado del panel de filtros. Todos opcionales."""
    # Rango de cuartil aceptado (numérico Q1<Q2<Q3<Q4).
    # best_quartile = mejor cuartil permitido (excluye los superiores si lo subes)
    # worst_quartile = peor cuartil permitido (excluye los inferiores)
    best_quartile: str = "Q1"                 # "Q1" | "Q2" | "Q3" | "Q4"
    worst_quartile: str = "Q4"                # "Q1" | "Q2" | "Q3" | "Q4"
    min_if: float = 0.0
    max_if: float | None = None               # tope superior de IF (None = sin tope)
    max_apc_eur: float | None = None
    accept_only_no_apc: bool = False
    accepted_oa_models: tuple[str, ...] = OA_MODELS_ALL  # qué modelos aceptas
    max_time_to_first_decision_weeks: int | None = None
    max_time_to_publication_weeks: int | None = None
    min_acceptance_rate_pct: float | None = None
    excluded_publishers: list[str] = field(default_factory=list)
    accepted_jcr_categories: list[str] = field(default_factory=list)  # vacío = sin filtro
    require_in_jcr: bool = True


# ── Resolución de campos con fallback entre fuentes ──────────────────────

def _resolve_quartile(row: pd.Series) -> str | None:
    """Cuartil efectivo: override manual del usuario > JCR oficial."""
    ov = row.get("quartile")
    if ov is not None and not pd.isna(ov) and str(ov).startswith("Q"):
        return str(ov)
    return row.get("jcr_quartile")


def _resolve_if(row: pd.Series) -> float:
    """IF efectivo: override manual > JCR JIF > proxy OpenAlex."""
    val = row.get("impact_factor")           # override manual
    if val is None or pd.isna(val):
        val = row.get("jcr_jif")
    if val is None or pd.isna(val):
        val = row.get("openalex_2yr_mean_citedness")
    try:
        return float(val) if val is not None and not pd.isna(val) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve_apc(row: pd.Series) -> float | None:
    """Prefiere el APC verificado por el usuario, luego DOAJ, luego OpenAlex."""
    apc = row.get("apc_eur")  # ya viene del override si está
    if apc is not None and not pd.isna(apc):
        return float(apc)
    apc = row.get("doaj_apc_eur")
    if apc is not None and not pd.isna(apc):
        return float(apc)
    return None


def _resolve_oa_model(row: pd.Series) -> str:
    """Modelo OA verificado > DOAJ. Si nada, 'unknown'."""
    model = row.get("oa_model")
    if model and isinstance(model, str) and model in OA_MODELS_ALL:
        return model
    return "unknown"


def _resolve_time_first_decision(row: pd.Series) -> float | None:
    """Tiempo a primera decisión: solo el verificado manualmente cuenta."""
    v = row.get("time_to_first_decision_weeks")
    if v is not None and not pd.isna(v):
        return float(v)
    return None


def _resolve_time_publication(row: pd.Series) -> float | None:
    """Tiempo de publicación: override > cálculo CrossRef."""
    v = row.get("time_to_publication_weeks_median")
    if v is not None and not pd.isna(v):
        return float(v)
    return None


# ── Aplicación de filtros ────────────────────────────────────────────────

def apply_filters(journals: pd.DataFrame, filters: JournalFilters) -> pd.DataFrame:
    """Devuelve un sub-DataFrame con las revistas que pasan los filtros.

    Política: permisivo con los huecos — una revista con dato faltante no
    se descarta a no ser que el filtro sea explícito.
    """
    if journals.empty:
        return journals

    df = journals.copy()

    # Rango de cuartil [mejor, peor] (numérico Q1=1 … Q4=4)
    best_rank = QUARTILE_ORDER.get(filters.best_quartile, 1)
    worst_rank = QUARTILE_ORDER.get(filters.worst_quartile, 4)
    if best_rank > 1 or worst_rank < 4:
        def _ok_q(r):
            q = _resolve_quartile(r)
            if q is None or pd.isna(q):
                return True  # permisivo con huecos
            rank = QUARTILE_ORDER.get(str(q), 99)
            return best_rank <= rank <= worst_rank
        df = df[df.apply(_ok_q, axis=1)]

    # IF mínimo
    if filters.min_if > 0:
        df = df[df.apply(lambda r: _resolve_if(r) >= filters.min_if, axis=1)]

    # IF máximo (tope superior)
    if filters.max_if is not None and filters.max_if > 0:
        def _ok_if_max(r):
            v = _resolve_if(r)
            return v == 0.0 or v <= filters.max_if  # 0.0 = sin dato → permisivo
        df = df[df.apply(_ok_if_max, axis=1)]

    # Modelos OA aceptados
    if set(filters.accepted_oa_models) != set(OA_MODELS_ALL):
        df = df[df.apply(lambda r: _resolve_oa_model(r) in filters.accepted_oa_models, axis=1)]

    # APC
    if filters.accept_only_no_apc:
        df = df[df.apply(lambda r: (_resolve_apc(r) is None or _resolve_apc(r) == 0), axis=1)]
    elif filters.max_apc_eur is not None:
        def _apc_ok(r):
            apc = _resolve_apc(r)
            return apc is None or apc <= filters.max_apc_eur
        df = df[df.apply(_apc_ok, axis=1)]

    # Tiempo a primera decisión
    if filters.max_time_to_first_decision_weeks is not None:
        def _ok_tfd(r):
            v = _resolve_time_first_decision(r)
            return v is None or v <= filters.max_time_to_first_decision_weeks
        df = df[df.apply(_ok_tfd, axis=1)]

    # Tiempo a publicación
    if filters.max_time_to_publication_weeks is not None:
        def _ok_tp(r):
            v = _resolve_time_publication(r)
            return v is None or v <= filters.max_time_to_publication_weeks
        df = df[df.apply(_ok_tp, axis=1)]

    # Tasa de aceptación mínima (solo aplica si la conocemos)
    if filters.min_acceptance_rate_pct is not None:
        def _ok_acc(r):
            v = r.get("acceptance_rate_pct")
            if v is None or pd.isna(v):
                return True
            return float(v) >= filters.min_acceptance_rate_pct
        df = df[df.apply(_ok_acc, axis=1)]

    # Editoriales excluidas
    if filters.excluded_publishers and "publisher" in df.columns:
        excluded = {p.upper() for p in filters.excluded_publishers}
        df = df[~df["publisher"].fillna("").str.upper().isin(excluded)]

    # Categorías JCR aceptadas (multiselect)
    if filters.accepted_jcr_categories and "jcr_categories_list" in df.columns:
        accepted = {c.upper() for c in filters.accepted_jcr_categories}
        def _has_accepted_cat(cats):
            if not isinstance(cats, list) or not cats:
                return False
            return any(str(c).upper() in accepted for c in cats)
        df = df[df["jcr_categories_list"].apply(_has_accepted_cat)]

    # En JCR
    if filters.require_in_jcr and "jcr_jif" in df.columns:
        # Si tenemos JCR cargado, exigir que la revista esté ahí
        df = df[df["jcr_jif"].notna()]

    return df.reset_index(drop=True)
