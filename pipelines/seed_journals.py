"""Lista semilla de revistas. Se parte del Excel del usuario y se amplía a
otras revistas relevantes en neurología, neurorrehabilitación, ictus,
estimulación cerebral y neuroimagen.

El pipeline `fetch_openalex.py` enriquecerá cada entrada con su ISSN real,
ID de OpenAlex, métricas de citas y artículos recientes.
"""
from __future__ import annotations

# (nombre canónico, ISSN si lo sabemos a priori (opcional), notas)
SEED_JOURNALS: list[tuple[str, str | None, str | None]] = [
    # ── Las 16 del Excel original ──
    ("Neurorehabilitation and Neural Repair", "1545-9683", "Q1 — favorita"),
    ("Topics in Stroke Rehabilitation", "1074-9357", "Q1 — favorita"),
    ("Brain", "0006-8950", "Q1 — favorita"),
    ("Brain Stimulation", "1935-861X", "Q1 — favorita"),
    ("Nature Neuroscience", "1097-6256", "Q1 — favorita"),
    ("Journal of Neurology", "0340-5354", "Q1 — favorita"),
    ("NeuroRehabilitation", "1053-8135", "Q2 — favorita"),
    ("Neurology International", "2035-8385", "Q2 — favorita"),
    ("Neural Plasticity", "2090-5904", "Q2 — favorita"),
    ("Frontiers in Neurology", "1664-2295", "Q2 — favorita"),
    ("Brain and Behavior", "2162-3279", "Q2 — favorita"),
    ("Annals of Neurology", "0364-5134", "Q1 — favorita"),
    ("Journal of Neurologic Physical Therapy", "1557-0576", "Q1 — favorita"),
    ("Stroke", "0039-2499", "Q1 — favorita"),
    ("Journal of Physiotherapy", "1836-9553", "Q1 — favorita"),
    ("Neuron", "0896-6273", "Q1 — favorita"),

    # ── Ampliación: neurorrehabilitación e ictus ──
    ("Clinical Rehabilitation", "0269-2155", None),
    ("Archives of Physical Medicine and Rehabilitation", "0003-9993", None),
    ("Journal of NeuroEngineering and Rehabilitation", "1743-0003", None),
    ("Disability and Rehabilitation", "0963-8288", None),
    ("Physiotherapy", "0031-9406", None),
    ("Physical Therapy", "0031-9023", None),
    ("International Journal of Stroke", "1747-4930", None),
    ("Cerebrovascular Diseases", "1015-9770", None),
    ("Journal of Stroke and Cerebrovascular Diseases", "1052-3057", None),
    ("Journal of Stroke", "2287-6391", None),
    ("Frontiers in Stroke", "2813-3056", None),

    # ── Estimulación cerebral / Neurociencia clínica ──
    ("Clinical Neurophysiology", "1388-2457", None),
    ("Neurophysiologie Clinique", "0987-7053", None),
    ("Brain Connectivity", "2158-0014", None),
    ("Neuromodulation", "1094-7159", None),
    ("Cortex", "0010-9452", None),
    ("Brain Sciences", "2076-3425", None),
    ("Journal of Neural Engineering", "1741-2560", None),

    # ── Neurología clínica general ──
    ("Lancet Neurology", "1474-4422", None),
    ("Neurology", "0028-3878", None),
    ("JAMA Neurology", "2168-6149", None),
    ("European Journal of Neurology", "1351-5101", None),
    ("Multiple Sclerosis Journal", "1352-4585", None),
    ("Movement Disorders", "0885-3185", None),
    ("Parkinsonism & Related Disorders", "1353-8020", None),
    ("Epilepsia", "0013-9580", None),
    ("Journal of the Neurological Sciences", "0022-510X", None),
    ("BMC Neurology", "1471-2377", None),
    ("Acta Neurologica Scandinavica", "0001-6314", None),

    # ── Neuroimagen ──
    ("NeuroImage", "1053-8119", None),
    ("Human Brain Mapping", "1065-9471", None),
    ("NeuroImage: Clinical", "2213-1582", None),
    ("Journal of Cerebral Blood Flow and Metabolism", "0271-678X", None),
    ("American Journal of Neuroradiology", "0195-6108", None),
    ("Neuroradiology", "0028-3940", None),

    # ── Neurociencia básica + cognitiva ──
    ("Nature Reviews Neuroscience", "1471-003X", None),
    ("Trends in Neurosciences", "0166-2236", None),
    ("Annual Review of Neuroscience", "0147-006X", None),
    ("Cerebral Cortex", "1047-3211", None),
    ("Journal of Neuroscience", "0270-6474", None),
    ("Progress in Neurobiology", "0301-0082", None),
    ("Neuroscience & Biobehavioral Reviews", "0149-7634", None),
    ("Behavioural Brain Research", "0166-4328", None),
]


def favorites_issn() -> set[str]:
    """ISSNs de las 16 revistas favoritas (las que estaban en el Excel)."""
    return {issn for _name, issn, note in SEED_JOURNALS if note and "favorita" in note and issn}
