# 📚 Buscador de Normativa Educativa

Aplicación Streamlit para consultar **38 documentos oficiales de normativa educativa** de Castilla y León mediante búsqueda semántica. Las respuestas se generan **exclusivamente** a partir del contenido de los PDF, usando el modelo **llama-3.3-70b-versatile** de Groq.

---

## 🗂 Estructura del repositorio

```
📁 tu-repositorio/
├── app.py                  ← Aplicación Streamlit
├── preprocess.py           ← Script de preprocesamiento (ejecutar 1 vez)
├── requirements.txt
├── faiss_index.bin         ← Índice semántico (generado por preprocess.py)
├── chunks_metadata.json    ← Metadatos de fragmentos (generado por preprocess.py)
├── .gitignore
├── README.md
└── Normativa_Oficial/      ← Carpeta con los 38 PDFs
    ├── 01_Ley_Organica_2_2006_LOE_consolidada.pdf
    ├── 02_Ley_Organica_3_2020_LOMLOE.pdf
    └── ... (38 archivos)
```

---

## ⚙️ Puesta en marcha (paso a paso)

### 1. Clona el repositorio y coloca los PDFs

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

Copia los 38 archivos PDF en la carpeta `Normativa_Oficial/`.

### 2. Crea un entorno virtual e instala dependencias

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configura la URL de tu repositorio GitHub

Abre `app.py` y edita la línea:

```python
GITHUB_REPO = "https://github.com/TU_USUARIO/TU_REPOSITORIO"
```

Sustituyendo `TU_USUARIO` y `TU_REPOSITORIO` por los valores reales.

### 4. Genera el índice semántico (⚠️ solo una vez)

```bash
python preprocess.py
```

Este proceso:
- Lee los 38 PDFs página a página
- Divide el texto en fragmentos solapados
- Calcula embeddings semánticos con `paraphrase-multilingual-mpnet-base-v2`
- Guarda `faiss_index.bin` y `chunks_metadata.json`

⏱️ **Tiempo estimado:** 5–15 minutos según tu CPU.

### 5. Sube todo a GitHub

```bash
git add .
git commit -m "Add PDFs, index and Streamlit app"
git push
```

> ℹ️ `faiss_index.bin` puede ser grande. Si supera los 100 MB, usa [Git LFS](https://git-lfs.github.com/).

### 6. Despliega en Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
2. Pulsa **"New app"** y selecciona tu repositorio.
3. Archivo principal: `app.py`.
4. Haz clic en **Deploy**.

---

## 🔑 Groq API Key

Necesitas una clave gratuita de Groq para que la app genere respuestas.

1. Regístrate en [console.groq.com](https://console.groq.com)
2. Ve a **API Keys** → **Create API Key**
3. Introduce la clave en el panel lateral de la app al usarla.

---

## 🧠 Cómo funciona

```
Pregunta del usuario
        │
        ▼
Embedding semántico (sentence-transformers)
        │
        ▼
Búsqueda en índice FAISS (top-6 fragmentos más similares)
        │
        ▼
Contexto → Groq llama-3.3-70b-versatile
        │
        ▼
Respuesta + fuentes con enlace y página
```

La IA recibe **solo** los fragmentos recuperados y tiene instrucciones estrictas de no inventar información.

---

## 📋 Normativa incluida

| # | Documento |
|---|-----------|
| 01 | LOE – Ley Orgánica 2/2006 de Educación (consolidada) |
| 02 | LOMLOE – Ley Orgánica 3/2020 |
| 03 | Decreto 52/2018 – Admisión |
| 04 | Decreto 32/2021 – Modifica Decreto 52/2018 |
| 05 | Orden EDU/70/2019 – Admisión |
| 06 | Orden EDU/1623/2021 – Modifica Orden EDU/70/2019 |
| 07 | Resolución 26/01/2026 – Admisión 2.º ciclo Infantil y Primaria |
| 08 | Orden 07/02/2001 – Jornada Escolar |
| 09 | Orden EDU/1766/2003 – Modifica Jornada Escolar |
| 10 | Orden EDU/20/2014 – Modifica Jornada Escolar |
| 11 | Orden EDU/13/2015 – Inspección Educativa |
| 12 | Real Decreto 95/2022 – Ordenación Educación Infantil |
| 13 | Decreto 37/2022 – Currículo Infantil CyL |
| 14 | Decreto 12/2008 – Primer Ciclo Infantil |
| 15 | Orden EDU/904/2011 – Desarrolla Decreto 12/2008 |
| 16 | Orden EDU/1511/2023 – Modifica Orden EDU/904/2011 |
| 17 | Orden EDU/95/2022 – Admisión Primer Ciclo Infantil |
| 18 | Orden EDU/117/2023 – Modifica Orden EDU/95/2022 |
| 19 | Resolución 26/01/2026 – Admisión Primer Ciclo Infantil |
| 20 | Orden EDU/1063/2022 – Calendario y Horario Primer Ciclo |
| 21 | Decreto 11/2023 – Precios Públicos Primer Ciclo |
| 22 | Decreto 17/2024 – Modifica Decreto 11/2023 |
| 23 | Orden EDU/593/2018 – Permanencia NEE en Infantil |
| 24 | Real Decreto 157/2022 – Ordenación Educación Primaria |
| 25 | Decreto 38/2022 – Currículo Primaria CyL |
| 26 | Orden EDU/423/2024 – Evaluación y Promoción Primaria |
| 27 | Orden EDU/17/2024 – Evaluación de Diagnóstico |
| 28 | Orden EDU/286/2016 – Vigencia Libros de Texto |
| 29 | Decreto 3/2019 – RELEO Plus |
| 30 | Orden EDU/167/2019 – RELEO Plus (bases) |
| 31 | Orden EDU/49/2020 – Modifica Orden EDU/167/2019 |
| 32 | Orden EDU/1861/2022 – Mejora del Éxito Educativo |
| 33 | Orden EDU/1152/2010 – Respuesta Educativa NEAE |
| 34 | Orden EDU/371/2018 – Modifica Orden EDU/1152/2010 |
| 35 | Resolución 17/08/2009 – Adaptaciones Curriculares Significativas |
| 36 | Orden EDU/865/2009 – Evaluación NEE |
| 37 | Orden EDU/1865/2004 – Flexibilización Alumnado Superdotado |
| 38 | Orden EDU/641/2012 – Prácticum Grados Infantil y Primaria |
