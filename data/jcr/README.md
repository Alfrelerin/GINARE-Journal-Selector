# Carpeta de exportaciones de JCR (Clarivate)

Esta carpeta está pensada para que **cada usuario** coloque aquí sus propios CSV/XLSX exportados desde Journal Citation Reports (JCR) con su acceso institucional (en tu caso, vía la UAM).

**No se sube nada de aquí al repositorio** (está excluido en `.gitignore`) porque los datos de Clarivate son propietarios y su licencia no permite redistribución pública.

## Cómo descargar tus CSV de JCR (vía UAM)

1. Entra a Web of Science a través del proxy de la UAM (biblioteca → bases de datos → Web of Science).
2. Una vez dentro, pulsa en "Products" → "Journal Citation Reports (JCR)".
3. Pestaña "Categories" → busca y abre cada una de tus categorías de interés:
   - Clinical Neurology
   - Neurosciences
   - Rehabilitation
   - Radiology, Nuclear Medicine & Medical Imaging
4. En cada categoría, pulsa "Export" (icono arriba a la derecha) → elige **CSV** o **XLS** → marca "All journals" o las columnas que quieras (al menos: *Journal name, ISSN, eISSN, JIF, JIF Quartile, JCI, Category*).
5. Guarda los archivos en esta carpeta con nombres descriptivos, por ejemplo:
   - `jcr_clinical_neurology_2024.csv`
   - `jcr_neurosciences_2024.csv`
   - `jcr_rehabilitation_2024.csv`
   - `jcr_radiology_neuroimaging_2024.csv`

Después, ejecuta el pipeline de unificación:

```bash
python pipelines/build_jcr_list.py
```

Esto leerá todos los CSV de esta carpeta y construirá `data/jcr_index.parquet`, que la app cruzará con los datos de OpenAlex y SJR para mostrarte el IF y el cuartil oficial de Clarivate.

## Si no tienes acceso a JCR

La app sigue funcionando sin esta carpeta: usará el **SJR (Scimago Journal Rank)** como proxy de cuartil y la métrica **2-year mean citedness de OpenAlex** como proxy del IF. La calidad de la recomendación es prácticamente equivalente; solo perderás el IF "oficial".
