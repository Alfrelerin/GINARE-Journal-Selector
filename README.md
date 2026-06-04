# 📚 Recomendador de revistas para envío de artículos

Aplicación web local que recomienda **revistas científicas candidatas** para
publicar un artículo, a partir del **título + abstract** del manuscrito.

Pensada para investigadores en **neurología, neurorrehabilitación, ictus,
estimulación cerebral y neuroimagen**, pero el catálogo de revistas y las
categorías JCR son configurables.

---

## ¿Cómo funciona?

1. **Indiza** miles de revistas indexadas en JCR (Master Journal List de
   Clarivate) y descarga sus metadatos abiertos (OpenAlex + Scimago SJR + DOAJ).
2. Para cada revista, descarga los **últimos ~50–200 artículos** publicados.
3. **Embebe** cada artículo con **SPECTER** (modelo de Allen AI entrenado para
   literatura científica).
4. Cuando le pasas tu título + abstract, embebe tu manuscrito y calcula la
   **similitud coseno** contra el corpus de cada revista, agregando con la
   media de los top-k más parecidos.
5. Combina esa afinidad temática con tus filtros y prioridades (cuartil, IF,
   APC, OA, tiempo de revisión, editorial vetada) para producir un **ranking
   final** con justificación.
6. Te muestra, para cada revista, los **artículos previos más parecidos al
   tuyo** — un argumento sólido para defender el *scope match* en la cover
   letter.

---

## Instalación rápida

### Requisitos
- macOS, Linux o Windows
- Conexión a internet (la primera vez descarga ~500 MB de modelo + datos)
- ~2 GB de espacio en disco

### Mac / Linux
```bash
git clone https://github.com/<tu-usuario>/app-eleccion-revistas.git
cd app-eleccion-revistas
bash setup.sh
source .venv/bin/activate
streamlit run app.py
```

### Windows
```cmd
git clone https://github.com/<tu-usuario>/app-eleccion-revistas.git
cd app-eleccion-revistas
setup.bat
.venv\Scripts\activate
streamlit run app.py
```

El primer arranque abre `http://localhost:8501`. La app ya viene con un
**mini-índice precomputado** de **59 revistas** (16 favoritas del autor +
expansión a las principales del campo) y **1.995 artículos**. Para tener el
ranking semántico necesitas calcular los embeddings (un único comando, ver
abajo).

---

## Construcción del índice completo

Cuando quieras tener el ranking semántico funcionando, o ampliar el universo
de revistas, corre:

```bash
python -m pipelines.run_all
```

Esto ejecuta en cadena 7 pipelines:

| Paso | Pipeline | Qué hace | Tiempo aprox. |
| --- | --- | --- | --- |
| 1 | `fetch_openalex.py` | Resuelve cada revista en OpenAlex, descarga metadatos y artículos | 1-5 min |
| 2 | `fetch_sjr.py` | Ranking SJR por categorías (cuartil abierto) | 10 seg |
| 3 | `build_jcr_list.py` | Lee tus CSV de JCR de `data/jcr/` y construye índice + posición por categoría (2/231) | <1 min |
| 4 | `fetch_doaj.py` | Modelo OA autoritativo (Diamond / Gold / Hybrid / Subscription) + APC oficial | 1-2 min |
| 5 | `fetch_sherpa.py` | Política de auto-archivo (preprint/postprint/embargo). Necesita key gratis de Sherpa Romeo | 1-2 min |
| 6 | `compute_review_times.py` | Tiempos mediana de revisión desde CrossRef (received → published) | 3-5 min |
| 7 | `compute_embeddings.py` | Embeddings SPECTER del corpus | **10-90 min la primera vez** |

Para activar Sherpa Romeo (paso 5), regístrate en https://v2.sherpa.ac.uk/cgi/register
y crea un fichero `.env` en la raíz del proyecto con:

```
SHERPA_API_KEY=tu_key_aqui
```

Sin ese fichero, el pipeline se salta y la app sigue funcionando.

Los pipelines son idempotentes y cachean en `data/openalex/cache/` para evitar
re-descargas. Puedes saltar pasos con flags:

```bash
python -m pipelines.run_all --skip-embeddings   # solo metadatos
python -m pipelines.run_all --skip-sjr          # ya cacheado
```

### Añadir revistas al catálogo

Edita `pipelines/seed_journals.py` y añade tuplas `(nombre, issn, nota)`.
Re-corre `python -m pipelines.run_all`.

### Conectar tus datos de JCR (Clarivate)

JCR es propietario y por licencia **no puede redistribuirse** públicamente,
pero **cada usuario puede descargar sus CSVs** con su cuenta institucional
(en mi caso, vía UAM) y colocarlos en `data/jcr/`. La app detectará los
ficheros y enriquecerá el ranking con el **IF y el cuartil oficial**.

Pasos detallados → [`data/jcr/README.md`](data/jcr/README.md).

---

## Estructura del proyecto

```
app-eleccion-revistas/
├── app.py                       # UI Streamlit
├── core/
│   ├── config.py                # rutas, categorías JCR objetivo, constantes
│   ├── data_loader.py           # carga el índice (revistas + artículos + embeddings)
│   ├── filters.py               # filtros por cuartil, IF, APC, OA, …
│   ├── recommender.py           # SPECTER + ranking + scoring ponderado
│   └── exporter.py              # exporta a Excel con el formato original
├── pipelines/
│   ├── seed_journals.py         # lista semilla de revistas
│   ├── fetch_openalex.py        # descarga metadatos + artículos
│   ├── fetch_sjr.py             # descarga ranking SJR
│   ├── build_jcr_list.py        # parsea tus CSV de JCR locales
│   ├── compute_embeddings.py    # SPECTER → embeddings.npy
│   └── run_all.py               # orquestador
├── data/
│   ├── journals.parquet         # índice de revistas (incluido en el repo)
│   ├── articles.parquet         # corpus de artículos (incluido)
│   ├── favorites.json           # ISSNs marcados como favoritos
│   ├── embeddings.npy           # se genera con compute_embeddings.py
│   ├── jcr/                     # ← coloca aquí tus CSV de JCR (no se sube)
│   └── sjr/                     # CSVs de Scimago
├── requirements.txt
├── pyproject.toml
├── setup.sh / setup.bat
└── README.md
```

---

## Fuentes de datos y licencias

| Fuente | Qué aporta | Licencia | ¿Se redistribuye? |
| --- | --- | --- | --- |
| [OpenAlex](https://openalex.org) | Metadatos de revistas, ISSN, citas, artículos, abstracts (índice invertido) | CC0 | ✅ |
| [Scimago SJR](https://www.scimagojr.com) | Cuartil + h-index por categoría | CC BY-NC | ✅ (no comercial) |
| [DOAJ](https://doaj.org) | Open Access, APC | CC BY-SA | ✅ |
| [Clarivate JCR](https://jcr.clarivate.com) | IF oficial, cuartil oficial JCR | Propietario | ❌ — cada usuario aporta los suyos en `data/jcr/` |
| [Master Journal List](https://mjl.clarivate.com) | Listado de ISSNs en WoS | Cuenta gratis | ❌ — descárgalo tú |
| [SPECTER (allenai)](https://github.com/allenai/specter) | Modelo de embeddings | Apache 2.0 | ✅ |

---

## Sistema de verificación manual

Cada revista en la app tiene un botón **"✏️ Editar / verificar"**. Puedes
fijar el modelo OA real, APC actualizado, semanas a primera decisión, tasa de
aceptación, URL correcta y notas libres. Al guardar marca un **✅ verificado
el DD/MM/AAAA** que persiste en `data/overrides.json`.

Como `overrides.json` se sube al repo público, las verificaciones son
**compartidas** — si otros investigadores claman el repo se benefician de
tus comprobaciones y viceversa.

## Roadmap (ideas para v2)

- [ ] Co-citación: usar las referencias de tu artículo para reforzar el matching.
- [ ] Alertas: aviso cuando una revista de tu lista publique algo muy similar a tu línea.
- [ ] Historial: guardar artículos enviados y respuestas (aceptado, rechazado, R&R).
- [ ] Detección de revistas predatorias (cruce con Cabells / Beall's list).
- [ ] Cambio a **SPECTER2** con adapters cuando se estabilice la integración.
- [ ] Despliegue opcional en Streamlit Cloud / Hugging Face Spaces para uso compartido.

---

## Cómo subir tu copia a GitHub

```bash
cd app-eleccion-revistas
git init
git add .
git commit -m "Primer commit del recomendador de revistas"
gh repo create app-eleccion-revistas --public --source=. --push
# o, si prefieres:
# git remote add origin https://github.com/<tu-usuario>/app-eleccion-revistas.git
# git push -u origin main
```

El `.gitignore` ya excluye los CSVs de JCR (propietarios) y los binarios pesados
generados localmente.

---

## Licencia

Código: MIT (ver `LICENSE`).
Los datos en `data/jcr/` (cuando existan) están sujetos a la licencia
propietaria de Clarivate y **no se redistribuyen**.
