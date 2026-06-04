"""Orquestador: corre los pipelines en orden.

Uso:
    python -m pipelines.run_all
"""
from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main(
    skip_embeddings: bool = False,
    skip_sjr: bool = False,
    skip_doaj: bool = False,
    skip_sherpa: bool = False,
    skip_review_times: bool = False,
) -> None:
    steps = [
        ("OpenAlex (metadatos + artículos)", "pipelines.fetch_openalex", False),
        ("SJR (Scimago, cuartiles abiertos)", "pipelines.fetch_sjr", skip_sjr),
        ("JCR (tus CSVs de data/jcr/, opcional)", "pipelines.build_jcr_list", False),
        ("DOAJ (modelo OA real + APC oficial)", "pipelines.fetch_doaj", skip_doaj),
        ("Sherpa Romeo (auto-archivo, opcional)", "pipelines.fetch_sherpa", skip_sherpa),
        ("Tiempos de publicación (CrossRef)", "pipelines.compute_review_times", skip_review_times),
        ("Embeddings SPECTER (largo, primera vez)", "pipelines.compute_embeddings", skip_embeddings),
    ]
    for i, (label, mod_name, skip) in enumerate(steps, start=1):
        if skip:
            log.info(" [%d/%d] %s … (saltado)", i, len(steps), label)
            continue
        log.info("=" * 60)
        log.info(" [%d/%d] %s", i, len(steps), label)
        log.info("=" * 60)
        import importlib
        mod = importlib.import_module(mod_name)
        # Algunos exportan build_index, otros main
        fn = getattr(mod, "build_index", None) or getattr(mod, "main", None)
        if fn is None:
            log.warning("  Saltado: %s no expone build_index() ni main()", mod_name)
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.error("  Error en %s: %s", mod_name, exc)
            log.info("  Continuando con el siguiente paso…")

    log.info("✔ Todo listo. Lanza la app con:  streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--skip-sjr", action="store_true")
    p.add_argument("--skip-doaj", action="store_true")
    p.add_argument("--skip-sherpa", action="store_true")
    p.add_argument("--skip-review-times", action="store_true")
    args = p.parse_args()
    main(
        skip_embeddings=args.skip_embeddings,
        skip_sjr=args.skip_sjr,
        skip_doaj=args.skip_doaj,
        skip_sherpa=args.skip_sherpa,
        skip_review_times=args.skip_review_times,
    )
