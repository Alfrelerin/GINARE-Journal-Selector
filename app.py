"""Streamlit UI del recomendador de revistas.

Lanzar con:
    streamlit run app.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import EMBEDDINGS_PATH
from core.data_loader import apply_uploaded_jcr, load_index, save_favorites
from core.exporter import recommendations_to_dataframe, to_excel_bytes
from core.filters import JournalFilters, OA_MODELS_ALL, apply_filters
from core.overrides import EDITABLE_FIELDS, OA_MODELS, upsert_override
from core.recommender import (
    RecommendationRequest,
    ScoringWeights,
    rank_journals,
)

APP_TITLE = "GINARE Journal Selector"
LOGO_PATH = str(Path(__file__).resolve().parent / "assets" / "ginare_logo.png")

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

# Logo de marca (arriba a la izquierda y en la barra lateral)
if os.path.exists(LOGO_PATH):
    try:
        st.logo(LOGO_PATH, size="large")
    except Exception:  # noqa: BLE001 — versiones antiguas de Streamlit
        pass


# ─────────────────────────────────────────────────────────────────────────
#  CACHÉ
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_index_cached():
    return load_index()


def _invalidate_index_cache():
    _load_index_cached.clear()


@st.cache_resource(show_spinner="Cargando modelo SPECTER (primera vez tarda ~1 minuto)…")
def _warmup_model():
    from core.recommender import _load_model
    _load_model()
    return True


# ─────────────────────────────────────────────────────────────────────────
#  COMPONENTES
# ─────────────────────────────────────────────────────────────────────────
def _badge(label: str, ok: bool, detail: str = "") -> str:
    icon = "🟢" if ok else "⚪"
    text = f"{icon} **{label}**"
    if detail:
        text += f" · {detail}"
    return text


def _render_header(index) -> None:
    s = index.sources
    cols = st.columns([2, 1, 1, 1, 1, 1])
    with cols[0]:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=240)
        else:
            st.markdown(f"### {APP_TITLE}")
    cols[1].markdown(_badge("OpenAlex", s.journals, f"{index.n_journals()}"))
    cols[2].markdown(_badge("JCR", s.jcr, "local" if s.jcr else "no"))
    cols[3].markdown(_badge("DOAJ", s.doaj))
    cols[4].markdown(_badge("Sherpa", s.sherpa))
    cols[5].markdown(_badge("Verificadas", s.overrides_count > 0, f"{s.overrides_count}"))


def _check_index(index) -> None:
    if index.n_journals() == 0:
        st.error(
            "**No hay índice de revistas.** Construye uno corriendo en terminal:\n\n"
            "```bash\npython -m pipelines.run_all\n```"
        )
        st.stop()
    if not EMBEDDINGS_PATH.exists():
        st.warning(
            "⚠️ **Faltan los embeddings**. Corre:\n\n"
            "```bash\npython -m pipelines.compute_embeddings\n```"
        )


def _render_sidebar_filters(index) -> tuple[JournalFilters, ScoringWeights, int]:
    with st.sidebar:
        st.markdown("### Filtros y prioridades")

        with st.expander("🎚️ Filtros", expanded=True):
            require_jcr = st.checkbox("Solo revistas en JCR", value=True,
                                     help="Si tu CSV de JCR no está cargado este filtro se desactiva.")

            st.markdown("**Rango de cuartil (JCR)**")
            cq1, cq2 = st.columns(2)
            quartiles = ["Q1", "Q2", "Q3", "Q4"]
            best_q = cq1.selectbox("Mejor", quartiles, index=0,
                                   help="Mejor cuartil que aceptas. Súbelo (p.ej. Q2) para "
                                        "excluir las más competitivas (Q1).")
            worst_q = cq2.selectbox("Peor", quartiles, index=3,
                                    help="Peor cuartil que aceptas. Para apuntar a gama media "
                                         "elige p.ej. Mejor=Q2 y Peor=Q3 → solo Q2 y Q3.")

            cif1, cif2 = st.columns(2)
            min_if = cif1.number_input("IF mínimo", 0.0, 100.0, 0.0, 0.5)
            max_if = cif2.number_input("IF máximo", 0.0, 100.0, 0.0, 0.5,
                                       help="0 = sin tope.")

            st.markdown("**Modelo Open Access aceptado**")
            oa_accept = []
            oa_labels = {
                "diamond": "Diamond (sin APC ni suscripción)",
                "gold": "Gold (OA con APC)",
                "hybrid": "Híbrido (paga OA opcional)",
                "subscription": "Suscripción tradicional",
                "unknown": "Desconocido",
            }
            for m in OA_MODELS_ALL:
                if st.checkbox(oa_labels[m], value=True, key=f"oa_{m}"):
                    oa_accept.append(m)

            max_apc = st.number_input("APC máx. (€)", 0, 10000, 0, 250, help="0 = sin límite")
            only_no_apc = st.checkbox("Solo gratis para autor (Diamond)", value=False)

            st.markdown("**Tiempos**")
            max_first = st.number_input("Máx. semanas a 1ª decisión", 0, 52, 0, 1,
                                       help="0 = sin límite. Aplica si está verificado manualmente.")
            max_pub = st.number_input("Máx. semanas a publicación", 0, 104, 0, 2,
                                     help="0 = sin límite. Calculado desde CrossRef.")

            min_acc = st.number_input("Tasa aceptación mínima (%)", 0.0, 100.0, 0.0, 5.0,
                                     help="0 = sin filtro. Aplica si está verificado.")

            publishers_all = sorted(index.journals["publisher"].dropna().unique().tolist()) if "publisher" in index.journals.columns else []
            excluded_pubs = st.multiselect("Editoriales a excluir", publishers_all)

            # Categorías JCR cargadas (solo si hay datos JCR)
            jcr_cats_all: list[str] = []
            if not index.jcr_ranks.empty and "jcr_category" in index.jcr_ranks.columns:
                jcr_cats_all = sorted(index.jcr_ranks["jcr_category"].dropna().unique().tolist())
            accepted_cats = st.multiselect(
                "Categorías JCR (vacío = todas)",
                jcr_cats_all,
                help="Limita el ranking a revistas indexadas en estas categorías JCR. "
                     "Útil si quieres p.ej. solo Rehabilitation + Clinical Neurology.",
            )

        with st.expander("⚖️ Pesos del ranking", expanded=False):
            w_topic = st.slider("Afinidad temática (SPECTER)", 0.0, 1.0, 0.65, 0.05)
            w_quart = st.slider("Cuartil", 0.0, 1.0, 0.15, 0.05)
            w_if = st.slider("Impact Factor", 0.0, 1.0, 0.10, 0.05)
            w_speed = st.slider("Velocidad", 0.0, 1.0, 0.05, 0.05)
            w_cost = st.slider("Coste (menor = mejor)", 0.0, 1.0, 0.05, 0.05)

        with st.expander("🎯 Resultados", expanded=False):
            top_n = st.slider("Cuántas revistas mostrar", 5, 50, 20, 5)

    filters = JournalFilters(
        best_quartile=best_q,
        worst_quartile=worst_q,
        min_if=min_if,
        max_if=max_if if max_if > 0 else None,
        max_apc_eur=max_apc if max_apc > 0 else None,
        accept_only_no_apc=only_no_apc,
        accepted_oa_models=tuple(oa_accept) if oa_accept else OA_MODELS_ALL,
        max_time_to_first_decision_weeks=max_first if max_first > 0 else None,
        max_time_to_publication_weeks=max_pub if max_pub > 0 else None,
        min_acceptance_rate_pct=min_acc if min_acc > 0 else None,
        excluded_publishers=excluded_pubs,
        accepted_jcr_categories=accepted_cats,
        require_in_jcr=require_jcr,
    )
    weights = ScoringWeights(w_topic, w_quart, w_if, w_speed, w_cost)
    return filters, weights, top_n


def _apc_label(apc_eur, oa_model: str | None) -> str:
    """Etiqueta honesta del APC: 'Gratis' solo si es realmente diamond/0;
    '—' cuando el dato falta (en vez de mostrar 'Gratis' engañosamente)."""
    if apc_eur is None:
        return "Gratis" if oa_model == "diamond" else "—"
    if apc_eur == 0:
        return "Gratis"
    return f"{apc_eur:.0f}"


def _oa_badge(model: str | None) -> str:
    return {
        "diamond": "💎 Diamond",
        "gold": "🥇 Gold OA",
        "hybrid": "🔀 Híbrido",
        "subscription": "🔒 Suscripción",
    }.get(model or "", "❔ Desconocido")


def _format_jcr_categories(cats: list[dict]) -> str:
    if not cats:
        return ""
    parts = []
    for c in cats:
        q = c.get("quartile") or "—"
        rank = c.get("rank")
        total = c.get("total")
        cat = c.get("category") or ""
        if rank and total:
            parts.append(f"{q} ({rank}/{total}) {cat}")
        elif rank:
            parts.append(f"{q} (#{rank}) {cat}")
        else:
            parts.append(f"{q} {cat}")
    return " · ".join(parts)


def _sync_overrides_to_github() -> None:
    """Sube data/overrides.json al repo de GitHub para que las verificaciones
    queden guardadas de forma permanente y compartida con todos los usuarios.

    Solo actúa si hay credenciales en los *secrets* de Streamlit:

        [github]
        token  = "ghp_xxx"                       # token con permiso de escritura
        repo   = "Alfrelerin/GINARE-Journal-Selector"
        branch = "main"

    Sin esos secrets (p.ej. en local) no hace nada: el mantenedor sube
    overrides.json a mano con un commit normal.
    """
    try:
        gh = st.secrets["github"]
        token = gh["token"]
        repo = gh["repo"]
        branch = gh.get("branch", "main")
    except Exception:  # noqa: BLE001 — sin secrets configurados → modo local
        return

    import base64
    import requests

    from core.config import OVERRIDES_PATH

    try:
        content = OVERRIDES_PATH.read_bytes()
    except OSError:
        return

    api = f"https://api.github.com/repos/{repo}/contents/data/overrides.json"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        # SHA del fichero actual (necesario para actualizarlo)
        sha = None
        r = requests.get(api, headers=headers, params={"ref": branch}, timeout=20)
        if r.status_code == 200:
            sha = r.json().get("sha")
        body = {
            "message": "Actualizar verificaciones (overrides) desde la app",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        pr = requests.put(api, headers=headers, json=body, timeout=20)
        if pr.status_code not in (200, 201):
            st.warning(f"No se pudo sincronizar con GitHub (código {pr.status_code}).")
    except requests.RequestException as exc:
        st.warning(f"No se pudo sincronizar con GitHub: {exc}")


def _render_verification_form(rec, journal_index) -> None:
    """Mini-formulario de verificación dentro del expander de una revista."""
    issn = rec.issn or ""
    if not issn:
        st.caption("Esta revista no tiene ISSN registrado, no se puede verificar.")
        return

    # Categoría JCR principal para precargar el formulario
    _q_ord = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    prim_cat = None
    if rec.jcr_categories:
        prim_cat = sorted(
            rec.jcr_categories,
            key=lambda c: (_q_ord.get(str(c.get("quartile") or ""), 99), c.get("rank") or 9999),
        )[0]

    key_prefix = f"verify_{issn}"
    with st.form(key=f"form_{key_prefix}"):
        st.markdown("**✏️ Editar / verificar datos**")
        c1, c2 = st.columns(2)
        with c1:
            oa = st.selectbox(
                "Modelo OA",
                OA_MODELS,
                index=OA_MODELS.index(rec.oa_model) if rec.oa_model in OA_MODELS else len(OA_MODELS) - 1,
                key=f"{key_prefix}_oa",
            )
            apc = st.number_input(
                "APC (€)",
                min_value=0.0, max_value=20000.0, step=50.0,
                value=float(rec.apc_eur) if rec.apc_eur else 0.0,
                key=f"{key_prefix}_apc",
            )
            home = st.text_input(
                "URL de la web de la revista (corregir si está mal)",
                value=rec.homepage_url or "",
                key=f"{key_prefix}_home",
            )
        with c2:
            t_first = st.number_input(
                "Semanas a 1ª decisión",
                min_value=0, max_value=52, step=1,
                value=int(rec.time_to_first_decision_weeks) if rec.time_to_first_decision_weeks else 0,
                key=f"{key_prefix}_first",
            )
            acc = st.number_input(
                "Tasa de aceptación (%)",
                min_value=0.0, max_value=100.0, step=1.0,
                value=float(rec.acceptance_rate_pct) if rec.acceptance_rate_pct else 0.0,
                key=f"{key_prefix}_acc",
            )

        st.markdown("**Datos bibliométricos** (rellena o corrige huecos)")
        b1, b2 = st.columns(2)
        with b1:
            publisher = st.text_input(
                "Editorial",
                value=rec.publisher or "",
                key=f"{key_prefix}_pub",
            )
            impact = st.number_input(
                "Impact Factor",
                min_value=0.0, max_value=1000.0, step=0.1,
                value=float(rec.impact_factor) if rec.impact_factor else 0.0,
                key=f"{key_prefix}_if",
            )
            q_opts = ["—", "Q1", "Q2", "Q3", "Q4"]
            quart = st.selectbox(
                "Cuartil",
                q_opts,
                index=q_opts.index(rec.quartile) if rec.quartile in q_opts else 0,
                key=f"{key_prefix}_quart",
            )
        with b2:
            cat_name = st.text_input(
                "Categoría JCR",
                value=(prim_cat.get("category") if prim_cat else "") or "",
                key=f"{key_prefix}_cat",
            )
            jrank = st.number_input(
                "Posición (rank)",
                min_value=0, max_value=100000, step=1,
                value=int(prim_cat["rank"]) if prim_cat and prim_cat.get("rank") else 0,
                key=f"{key_prefix}_jrank",
            )
            jtotal = st.number_input(
                "Total en la categoría",
                min_value=0, max_value=100000, step=1,
                value=int(prim_cat["total"]) if prim_cat and prim_cat.get("total") else 0,
                key=f"{key_prefix}_jtotal",
            )

        notes = st.text_area(
            "Notas (libre)",
            value="",
            key=f"{key_prefix}_notes",
            help="Editor responde rápido, tiende a pedir muchas revisiones, etc.",
        )

        c1, c2, c3 = st.columns(3)
        verify_all = c1.form_submit_button("✅ Guardar y marcar verificado")
        save_only = c2.form_submit_button("💾 Guardar sin verificar")
        cancel = c3.form_submit_button("✖ Cancelar")

        if verify_all or save_only:
            fields = {
                "oa_model": oa if oa else None,
                "apc_eur": float(apc) if apc > 0 else None,
                "time_to_first_decision_weeks": int(t_first) if t_first > 0 else None,
                "acceptance_rate_pct": float(acc) if acc > 0 else None,
                "homepage_url": home.strip() or None,
                "notes": notes.strip() or None,
                "publisher": publisher.strip() or None,
                "impact_factor": float(impact) if impact > 0 else None,
                "quartile": quart if quart in ("Q1", "Q2", "Q3", "Q4") else None,
                "manual_jcr_category": cat_name.strip() or None,
                "manual_jcr_rank": int(jrank) if jrank > 0 else None,
                "manual_jcr_total": int(jtotal) if jtotal > 0 else None,
            }
            verified = [k for k, v in fields.items() if v is not None] if verify_all else []
            upsert_override(issn=issn, fields=fields, verified_fields=verified)
            _invalidate_index_cache()
            # Sincroniza con GitHub para que la verificación quede guardada
            # para todos (si hay credenciales configuradas en los secrets).
            _sync_overrides_to_github()
            # Pedimos recalcular el ranking con los datos nuevos, manteniendo
            # la tabla visible (no se pierde como antes).
            st.session_state["force_recompute"] = True
            st.success("Cambios guardados. Actualizando…")
            st.rerun()


def _apply_jcr_uploads(index):
    """Cargador de JCR por sesión. Permite a cada usuario subir su propio
    jcr_index.parquet y jcr_ranks.parquet (datos Clarivate, licencia privada)
    sin que viajen en el repo. Se aplican solo en su sesión."""
    import io

    with st.sidebar:
        with st.expander("📥 Cargar mi JCR (Clarivate)", expanded=not index.sources.jcr):
            if index.sources.jcr:
                st.caption("✅ JCR cargado en esta sesión. Cuartiles, IF y rankings oficiales en uso.")
            else:
                st.caption(
                    "Sube tus ficheros JCR para usar cuartiles, IF y rankings oficiales. "
                    "**No se guardan en el servidor**: solo se usan en tu sesión "
                    "(la licencia de Clarivate no permite redistribuirlos)."
                )
            up_idx = st.file_uploader("jcr_index.parquet", type=["parquet"], key="up_jcr_index")
            up_ranks = st.file_uploader("jcr_ranks.parquet", type=["parquet"], key="up_jcr_ranks")

    def _read(uploaded, slot):
        """Lee el parquet subido, cacheándolo por sesión (solo re-lee si cambia)."""
        if uploaded is None:
            st.session_state.pop(f"{slot}_df", None)
            st.session_state.pop(f"{slot}_sig", None)
            return None
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get(f"{slot}_sig") != sig:
            try:
                st.session_state[f"{slot}_df"] = pd.read_parquet(io.BytesIO(uploaded.getvalue()))
                st.session_state[f"{slot}_sig"] = sig
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"No pude leer {uploaded.name}: {exc}")
                return None
        return st.session_state.get(f"{slot}_df")

    jcr_idx_df = _read(up_idx, "jcr_index")
    jcr_ranks_df = _read(up_ranks, "jcr_ranks")

    if jcr_idx_df is None and jcr_ranks_df is None:
        return index

    # Cacheamos el índice ya fusionado para no rehacer el merge en cada rerun.
    sig = (id(index),
           st.session_state.get("jcr_index_sig"),
           st.session_state.get("jcr_ranks_sig"))
    if st.session_state.get("_jcr_applied_sig") != sig:
        st.session_state["_jcr_applied_index"] = apply_uploaded_jcr(index, jcr_idx_df, jcr_ranks_df)
        st.session_state["_jcr_applied_sig"] = sig
    return st.session_state["_jcr_applied_index"]


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    index = _load_index_cached()
    index = _apply_jcr_uploads(index)
    _render_header(index)
    st.caption("Pega el título y abstract de tu artículo. La app puntúa por afinidad temática "
               "(SPECTER) y por los criterios que elijas.")
    _check_index(index)

    col_input, col_filters = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("Tu artículo")
        title = st.text_input("Título", key="title_input",
                             placeholder="Ej.: Transcranial direct current stimulation in subacute stroke rehabilitation")
        abstract = st.text_area("Abstract", height=240, key="abstract_input",
                               placeholder="Pega aquí el resumen completo (250-400 palabras suele dar el mejor matching).")
        run = st.button("🔍 Recomendar revistas", type="primary", use_container_width=True)

    with col_filters:
        st.subheader("Configuración")
        st.caption("Los filtros y pesos están en la barra lateral ←")
        if index.sources.jcr:
            st.success("✅ Tu JCR está cargado: cuartiles e IF oficiales en uso.")
        else:
            st.info("ℹ️ Sin JCR local. Usaremos el proxy de OpenAlex para el IF. "
                   "Coloca tus CSVs en `data/jcr/` y corre `python -m pipelines.build_jcr_list`.")

    filters, weights, top_n = _render_sidebar_filters(index)

    st.divider()

    # Recalculamos si se pulsa el botón o si una edición pidió refrescar
    # (force_recompute). Las recomendaciones se guardan en sesión para que la
    # tabla no desaparezca al marcar un favorito o editar una revista.
    force = st.session_state.pop("force_recompute", False)
    if run or force:
        if not title.strip() or not abstract.strip():
            if run:
                st.warning("Necesito al menos título + abstract para puntuar.")
        else:
            _warmup_model()
            with st.spinner("Calculando ranking…"):
                candidates = apply_filters(index.journals, filters)
                if candidates.empty:
                    st.error("Ningún candidato pasa los filtros. Relájalos un poco.")
                    st.session_state["recs"] = []
                else:
                    req = RecommendationRequest(title=title, abstract=abstract,
                                               weights=weights, max_results=top_n)
                    recs = rank_journals(
                        req, index,
                        candidate_journal_ids=candidates["journal_id"].tolist(),
                    )
                    if not recs:
                        st.warning("No se han generado recomendaciones. "
                                   "¿Tienes embeddings calculados?")
                    st.session_state["recs"] = recs

    stored_recs = st.session_state.get("recs")
    if stored_recs:
        _render_results(stored_recs, index)
    elif not (run or force):
        st.info("👈 Rellena título y abstract y pulsa **Recomendar revistas**.")


def _render_results(recs, index) -> None:
    st.subheader(f"🏆 Top {len(recs)} revistas recomendadas")

    _Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    def _primary_category(cats: list[dict]) -> dict | None:
        """Categoría 'principal': la de mejor cuartil y, a igualdad, mejor
        posición (rank más bajo)."""
        if not cats:
            return None
        def _key(c):
            q = _Q_ORDER.get(str(c.get("quartile") or ""), 99)
            rank = c.get("rank") or 9_999
            return (q, rank)
        return sorted(cats, key=_key)[0]

    def _split_jcr(cats: list[dict]) -> tuple[str, str, str]:
        """Devuelve (cuartil, ranking, categoría) de la categoría principal.
        Si hay varias categorías, añade '(+N)' al nombre."""
        prim = _primary_category(cats)
        if not prim:
            return "—", "—", "—"
        q = prim.get("quartile") or "—"
        rank = prim.get("rank")
        total = prim.get("total")
        ranking = f"{rank}/{total}" if rank and total else (f"#{rank}" if rank else "—")
        cat = (prim.get("category") or "—").title()
        extra = len(cats) - 1
        if extra > 0:
            cat += f" (+{extra})"
        return q, ranking, cat

    rows = []
    for i, r in enumerate(recs):
        q_col, rank_col, cat_col = _split_jcr(r.jcr_categories)
        rows.append({
        "★": "⭐" if (r.issn and r.issn in index.favorites) else "",
        "#": i + 1,
        "Revista": r.name,
        "Cuartil": q_col,
        "Ranking": rank_col,
        "Categoría": cat_col,
        "IF": f"{r.impact_factor:.2f}" if r.impact_factor else "—",
        "APC (€)": _apc_label(r.apc_eur, r.oa_model),
        "Modelo OA": _oa_badge(r.oa_model),
        "1ª decisión (sem)": f"{r.time_to_first_decision_weeks:.0f}" if r.time_to_first_decision_weeks else "—",
        "Publicación (sem)": f"{r.time_to_publication_weeks:.0f}" if r.time_to_publication_weeks else "—",
        "% aceptación": f"{r.acceptance_rate_pct:.0f}%" if r.acceptance_rate_pct else "—",
        "Editorial": (r.publisher or "")[:28],
        "Score": round(r.final_score, 3),
        "Afinidad": f"{r.topic_similarity:.0%}",
        "✓": "✅" if r.verified_fields else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        st.download_button("⬇️ Excel", data=to_excel_bytes(recs),
                          file_name="recomendaciones_revistas.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        csv = recommendations_to_dataframe(recs).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV", data=csv, file_name="recomendaciones.csv", mime="text/csv")

    st.divider()
    st.subheader("🔎 Detalle y verificación por revista")

    for i, r in enumerate(recs, start=1):
        title = f"**#{i} · {r.name}**  · Score {r.final_score:.3f}  · Afinidad {r.topic_similarity:.0%}"
        if r.issn and r.issn in index.favorites:
            title = "⭐ " + title
        if r.verified_fields:
            title += f"  · ✅ verificado el {r.verified_at}"

        with st.expander(title):
            top_cols = st.columns([2, 1, 1])
            with top_cols[0]:
                # Categorías JCR con posición
                if r.jcr_categories:
                    st.markdown("**Posición en JCR:** " + _format_jcr_categories(r.jcr_categories))
                else:
                    if r.quartile:
                        st.markdown(f"**Cuartil**: {r.quartile}")
            with top_cols[1]:
                if r.homepage_url:
                    st.link_button("🔗 Web de la revista", r.homepage_url, use_container_width=True)
                else:
                    st.caption("Web no disponible")
            with top_cols[2]:
                issn = r.issn or ""
                if issn:
                    fav_key = f"fav_{issn}"
                    is_fav = issn in index.favorites
                    new_fav = st.checkbox("⭐ Favorita", value=is_fav, key=fav_key)
                    if new_fav != is_fav:
                        if new_fav:
                            index.favorites.add(issn)
                        else:
                            index.favorites.discard(issn)
                        save_favorites(index.favorites)
                        # Mutamos el set en memoria (sin recargar todo el índice),
                        # así la tabla se mantiene. Refrescamos para que la
                        # estrella de la tabla se actualice al instante.
                        st.rerun()

            mcols = st.columns(6)
            mcols[0].metric("IF", f"{r.impact_factor:.2f}" if r.impact_factor else "—")
            _apc_m = _apc_label(r.apc_eur, r.oa_model)
            mcols[1].metric("APC", _apc_m if _apc_m in ("Gratis", "—") else f"{_apc_m} €")
            mcols[2].metric("OA", _oa_badge(r.oa_model))
            mcols[3].metric("1ª decisión",
                          f"{r.time_to_first_decision_weeks:.0f} sem" if r.time_to_first_decision_weeks else "—")
            mcols[4].metric("Publicación",
                          f"{r.time_to_publication_weeks:.0f} sem" if r.time_to_publication_weeks else "—")
            mcols[5].metric("Aceptación",
                          f"{r.acceptance_rate_pct:.0f}%" if r.acceptance_rate_pct else "—")

            st.markdown(
                f"**Desglose del score** · tema {r.topic_similarity:.0%} · "
                f"cuartil {r.quartile_score:.0%} · IF {r.impact_score:.0%} · "
                f"velocidad {r.speed_score:.0%} · coste {r.cost_score:.0%}"
            )

            if r.top_articles:
                st.markdown("**Artículos publicados ahí más parecidos al tuyo:**")
                for art in r.top_articles:
                    sim = art.get("similarity", 0)
                    year = art.get("year") or ""
                    t = art.get("title", "")
                    doi = art.get("doi")
                    if doi:
                        st.markdown(f"- [{t}]({doi}) · *{year}* · similitud {sim:.2f}")
                    else:
                        st.markdown(f"- {t} · *{year}* · similitud {sim:.2f}")

            with st.expander("✏️ Editar / verificar datos de esta revista", expanded=False):
                _render_verification_form(r, index)


if __name__ == "__main__":
    main()
